## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 79.4085105772
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612)
1: (-59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992)
2: (-53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439)
3: (-58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935)
4: (-66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492)
5: (-60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872)
6: (-95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498)
7: (-70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839)
8: (-77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492)
9: (-64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648)
10: (-95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122)
11: (-91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326)
12: (-91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001)
13: (-97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521)
14: (-141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317)
15: (-75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994)
16: (-96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474)
17: (-139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440)
18: (-90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867)
19: (-72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048)
20: (-67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908)
21: (-89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444)
22: (-90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815)
23: (-71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896)
24: (-86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379)
25: (-76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524)
26: (-101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566)
27: (-88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820)
28: (-71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006)
29: (-92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924)
30: (-88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199)
31: (-93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035)
32: (-93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315)
33: (-121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873)
34: (-100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062)
35: (-98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817)
36: (-100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760)
37: (-141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120)
38: (-119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850)
39: (-135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140)
40: (-112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437)
41: (-94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704)
42: (-69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776)

## BASE Result
execution time: IAR + LP analysis = 2.76 + 97.39 = 100.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -83.2047295, upper bound: 83.2047295


# Binary Search by BASE starts (time budget: 17899.85 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=124.9945068359375
rel_dist={1: [-79.47797671394284, 79.47797670615168]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=124.9945068359375
rel_dist={1: [-76.66888544603012, 76.66888544397699]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=124.9945068359375
rel_dist={1: [-77.72391643090478, 77.72391642674157]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=124.9945068359375
rel_dist={1: [-78.64399325943418, 78.64399325111157]}

## Binary Search Result
Binary search time: 518.03 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Relational Split (RS_dual_Z) starts
Time budget: 17381.81 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5010778, upper bound: 81.5540831
time: 146.02 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5540831, upper bound: 81.5010778
time: 97.89 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 244.06 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 244.06
Output dim: 1, lower bound: -81.5010778, upper bound: 81.5540831
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 244.06
Output dim: 1, lower bound: -81.5540831, upper bound: 81.5010778

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4559394, upper bound: 81.5526619
time: 95.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4996564, upper bound: 81.5044213
time: 102.63 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5044213, upper bound: 81.4996564
time: 102.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5526619, upper bound: 81.4559394
time: 137.78 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 242.92 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 242.92
Output dim: 1, lower bound: -81.4559394, upper bound: 81.5526619
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 242.92
Output dim: 1, lower bound: -81.4996564, upper bound: 81.5044213
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 242.92
Output dim: 1, lower bound: -81.5044213, upper bound: 81.4996564
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 242.92
Output dim: 1, lower bound: -81.5526619, upper bound: 81.4559394

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.3618199, upper bound: 81.4547093
time: 134.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.3581333, upper bound: 81.4584032
time: 147.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4053343, upper bound: 81.4066630
time: 107.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4016379, upper bound: 81.4103587
time: 229.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4103587, upper bound: 81.4016379
time: 96.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4066630, upper bound: 81.4053343
time: 119.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4584032, upper bound: 81.3581333
time: 154.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4547093, upper bound: 81.3618199
time: 112.44 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 268.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 268.97
Output dim: 1, lower bound: -81.3618199, upper bound: 81.4547093
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 268.97
Output dim: 1, lower bound: -81.3581333, upper bound: 81.4584032
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 268.97
Output dim: 1, lower bound: -81.4053343, upper bound: 81.4066630
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 268.97
Output dim: 1, lower bound: -81.4016379, upper bound: 81.4103587
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 268.97
Output dim: 1, lower bound: -81.4103587, upper bound: 81.4016379
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 268.97
Output dim: 1, lower bound: -81.4066630, upper bound: 81.4053343
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 268.97
Output dim: 1, lower bound: -81.4584032, upper bound: 81.3581333
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 268.97
Output dim: 1, lower bound: -81.4547093, upper bound: 81.3618199

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.3555122, upper bound: 81.3954375
time: 127.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.3249308, upper bound: 81.4511739
time: 113.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.3518356, upper bound: 81.3991404
time: 128.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.3212365, upper bound: 81.4548319
time: 91.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4019875, upper bound: 81.3733645
time: 116.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.3497435, upper bound: 81.3993633
time: 112.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.3983201, upper bound: 81.3770557
time: 101.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.3460521, upper bound: 81.4030562
time: 211.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4030562, upper bound: 81.3460521
time: 135.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.3770557, upper bound: 81.3983201
time: 103.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.3993633, upper bound: 81.3497435
time: 96.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.3733645, upper bound: 81.4019875
time: 161.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4548319, upper bound: 81.3212365
time: 90.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.3991404, upper bound: 81.3518356
time: 95.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4511739, upper bound: 81.3249308
time: 105.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.3954375, upper bound: 81.3555122
time: 170.38 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 278.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 278.60
Output dim: 1, lower bound: -81.3555122, upper bound: 81.3954375
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 278.60
Output dim: 1, lower bound: -81.3249308, upper bound: 81.4511739
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 278.60
Output dim: 1, lower bound: -81.3518356, upper bound: 81.3991404
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 278.60
Output dim: 1, lower bound: -81.3212365, upper bound: 81.4548319
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 278.60
Output dim: 1, lower bound: -81.4019875, upper bound: 81.3733645
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 278.60
Output dim: 1, lower bound: -81.3497435, upper bound: 81.3993633
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 278.60
Output dim: 1, lower bound: -81.3983201, upper bound: 81.3770557
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 278.60
Output dim: 1, lower bound: -81.3460521, upper bound: 81.4030562
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 278.60
Output dim: 1, lower bound: -81.4030562, upper bound: 81.3460521
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 278.60
Output dim: 1, lower bound: -81.3770557, upper bound: 81.3983201
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 278.60
Output dim: 1, lower bound: -81.3993633, upper bound: 81.3497435
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 278.60
Output dim: 1, lower bound: -81.3733645, upper bound: 81.4019875
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 278.60
Output dim: 1, lower bound: -81.4548319, upper bound: 81.3212365
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 278.60
Output dim: 1, lower bound: -81.3991404, upper bound: 81.3518356
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 278.60
Output dim: 1, lower bound: -81.4511739, upper bound: 81.3249308
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 278.60
Output dim: 1, lower bound: -81.3954375, upper bound: 81.3555122

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.2822856, upper bound: 81.3222091
time: 134.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.2822856, upper bound: 81.3929560
time: 124.92 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 262.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 262.16
Output dim: 1, lower bound: -81.2822856, upper bound: 81.3222091
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 262.16
Output dim: 1, lower bound: -81.2822856, upper bound: 81.3929560
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 262.16
Output dim: 1, lower bound: -81.3249308, upper bound: 81.4511739
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 262.16
Output dim: 1, lower bound: -81.3518356, upper bound: 81.3991404
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 262.16
Output dim: 1, lower bound: -81.3212365, upper bound: 81.4548319
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 262.16
Output dim: 1, lower bound: -81.4019875, upper bound: 81.3733645
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 262.16
Output dim: 1, lower bound: -81.3497435, upper bound: 81.3993633
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 262.16
Output dim: 1, lower bound: -81.3983201, upper bound: 81.3770557
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 262.16
Output dim: 1, lower bound: -81.3460521, upper bound: 81.4030562
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 262.16
Output dim: 1, lower bound: -81.4030562, upper bound: 81.3460521
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 262.16
Output dim: 1, lower bound: -81.3770557, upper bound: 81.3983201
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 262.16
Output dim: 1, lower bound: -81.3993633, upper bound: 81.3497435
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 262.16
Output dim: 1, lower bound: -81.3733645, upper bound: 81.4019875
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 262.16
Output dim: 1, lower bound: -81.4548319, upper bound: 81.3212365
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 262.16
Output dim: 1, lower bound: -81.3991404, upper bound: 81.3518356
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 262.16
Output dim: 1, lower bound: -81.4511739, upper bound: 81.3249308
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 262.16
Output dim: 1, lower bound: -81.3954375, upper bound: 81.3555122
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=124.9945068359375
rel_dist={1: [-81.5568719651244, 81.5568719651244]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1935852, upper bound: 80.2387948
time: 120.54 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2387948, upper bound: 80.1935852
time: 97.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 218.42 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 218.42
Output dim: 1, lower bound: -80.1935852, upper bound: 80.2387948
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 218.42
Output dim: 1, lower bound: -80.2387948, upper bound: 80.1935852

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1560070, upper bound: 80.2376545
time: 82.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1924439, upper bound: 80.1971410
time: 120.09 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1971410, upper bound: 80.1924439
time: 125.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2376545, upper bound: 80.1560070
time: 384.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 512.39 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 512.39
Output dim: 1, lower bound: -80.1560070, upper bound: 80.2376545
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 512.39
Output dim: 1, lower bound: -80.1924439, upper bound: 80.1971410
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 512.39
Output dim: 1, lower bound: -80.1971410, upper bound: 80.1924439
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 512.39
Output dim: 1, lower bound: -80.2376545, upper bound: 80.1560070

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.0843784, upper bound: 80.1603621
time: 120.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.0788948, upper bound: 80.1658324
time: 106.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1207572, upper bound: 80.1200781
time: 106.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1152775, upper bound: 80.1255651
time: 100.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1255651, upper bound: 80.1152775
time: 101.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1200781, upper bound: 80.1207572
time: 100.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1658324, upper bound: 80.0788948
time: 120.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1603621, upper bound: 80.0843784
time: 117.70 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 240.21 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 240.21
Output dim: 1, lower bound: -80.0843784, upper bound: 80.1603621
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 240.21
Output dim: 1, lower bound: -80.0788948, upper bound: 80.1658324
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 240.21
Output dim: 1, lower bound: -80.1207572, upper bound: 80.1200781
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 240.21
Output dim: 1, lower bound: -80.1152775, upper bound: 80.1255651
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 240.21
Output dim: 1, lower bound: -80.1255651, upper bound: 80.1152775
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 240.21
Output dim: 1, lower bound: -80.1200781, upper bound: 80.1207572
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 240.21
Output dim: 1, lower bound: -80.1658324, upper bound: 80.0788948
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 240.21
Output dim: 1, lower bound: -80.1603621, upper bound: 80.0843784

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.0795364, upper bound: 80.1093531
time: 105.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.0516548, upper bound: 80.1579747
time: 129.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.0461703, upper bound: 80.1148482
time: 257.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.0461703, upper bound: 80.1634194
time: 114.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1183584, upper bound: 80.0913153
time: 102.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.0726948, upper bound: 80.1142223
time: 106.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1128942, upper bound: 80.0968219
time: 170.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.0672064, upper bound: 80.1197086
time: 131.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1197086, upper bound: 80.0672064
time: 106.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.0968219, upper bound: 80.1128942
time: 105.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1142223, upper bound: 80.0726948
time: 119.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.0913153, upper bound: 80.1183584
time: 129.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1634194, upper bound: 80.0461703
time: 128.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1148482, upper bound: 80.0740562
time: 168.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1579747, upper bound: 80.0516548
time: 109.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1093531, upper bound: 80.0795364
time: 128.76 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 240.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 240.93
Output dim: 1, lower bound: -80.0795364, upper bound: 80.1093531
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 240.93
Output dim: 1, lower bound: -80.0516548, upper bound: 80.1579747
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 240.93
Output dim: 1, lower bound: -80.0461703, upper bound: 80.1148482
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 240.93
Output dim: 1, lower bound: -80.0461703, upper bound: 80.1634194
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 240.93
Output dim: 1, lower bound: -80.1183584, upper bound: 80.0913153
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 240.93
Output dim: 1, lower bound: -80.0726948, upper bound: 80.1142223
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 240.93
Output dim: 1, lower bound: -80.1128942, upper bound: 80.0968219
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 240.93
Output dim: 1, lower bound: -80.0672064, upper bound: 80.1197086
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 240.93
Output dim: 1, lower bound: -80.1197086, upper bound: 80.0672064
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 240.93
Output dim: 1, lower bound: -80.0968219, upper bound: 80.1128942
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 240.93
Output dim: 1, lower bound: -80.1142223, upper bound: 80.0726948
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 240.93
Output dim: 1, lower bound: -80.0913153, upper bound: 80.1183584
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 240.93
Output dim: 1, lower bound: -80.1634194, upper bound: 80.0461703
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 240.93
Output dim: 1, lower bound: -80.1148482, upper bound: 80.0740562
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 240.93
Output dim: 1, lower bound: -80.1579747, upper bound: 80.0516548
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 240.93
Output dim: 1, lower bound: -80.1093531, upper bound: 80.0795364
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=124.9945068359375
rel_dist={1: [-80.2411023357962, 80.2411023398264]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4350996, upper bound: 79.4761054
time: 281.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4761054, upper bound: 79.4350996
time: 144.06 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 425.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 425.57
Output dim: 1, lower bound: -79.4350996, upper bound: 79.4761054
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 425.57
Output dim: 1, lower bound: -79.4761054, upper bound: 79.4350996

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4011924, upper bound: 79.4751264
time: 115.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4341020, upper bound: 79.4390138
time: 93.24 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4390139, upper bound: 79.4341020
time: 183.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4751264, upper bound: 79.4011924
time: 116.06 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 302.04 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 302.04
Output dim: 1, lower bound: -79.4011924, upper bound: 79.4751264
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 302.04
Output dim: 1, lower bound: -79.4341020, upper bound: 79.4390138
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 302.04
Output dim: 1, lower bound: -79.4390139, upper bound: 79.4341020
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 302.04
Output dim: 1, lower bound: -79.4751264, upper bound: 79.4011924

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -79.3372460, upper bound: 79.4043505
time: 145.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.3302718, upper bound: 79.4113137
time: 130.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -79.3702469, upper bound: 79.3682712
time: 110.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -79.3632751, upper bound: 79.3752307
time: 90.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -79.3752307, upper bound: 79.3632751
time: 93.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -79.3682712, upper bound: 79.3702469
time: 106.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4113137, upper bound: 79.3302718
time: 117.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -79.4043505, upper bound: 79.3372460
time: 101.27 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 221.38 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 221.38
Output dim: 1, lower bound: -79.3372460, upper bound: 79.4043505
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 221.38
Output dim: 1, lower bound: -79.3302718, upper bound: 79.4113137
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 221.38
Output dim: 1, lower bound: -79.3702469, upper bound: 79.3682712
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 221.38
Output dim: 1, lower bound: -79.3632751, upper bound: 79.3752307
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 221.38
Output dim: 1, lower bound: -79.3752307, upper bound: 79.3632751
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 221.38
Output dim: 1, lower bound: -79.3682712, upper bound: 79.3702469
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 221.38
Output dim: 1, lower bound: -79.4113137, upper bound: 79.3302718
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 221.38
Output dim: 1, lower bound: -79.4043505, upper bound: 79.3372460

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -79.3259567, upper bound: 79.3653201
time: 131.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.3010017, upper bound: 79.4093707
time: 102.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4093707, upper bound: 79.3010017
time: 102.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -79.3653201, upper bound: 79.3259567
time: 114.67 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 219.31 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 219.31
Output dim: 1, lower bound: -79.3259567, upper bound: 79.3653201
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 219.31
Output dim: 1, lower bound: -79.3010017, upper bound: 79.4093707
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 219.31
Output dim: 1, lower bound: -79.4093707, upper bound: 79.3010017
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 219.31
Output dim: 1, lower bound: -79.3653201, upper bound: 79.3259567

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -79.2992963, upper bound: 79.3589439
time: 102.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -79.2505439, upper bound: 79.4076678
time: 102.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -79.4076678, upper bound: 79.2505439
time: 99.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -79.3589439, upper bound: 79.2992963
time: 111.87 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 213.83 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 213.83
Output dim: 1, lower bound: -79.2992963, upper bound: 79.3589439
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 213.83
Output dim: 1, lower bound: -79.2505439, upper bound: 79.4076678
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 213.83
Output dim: 1, lower bound: -79.4076678, upper bound: 79.2505439
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 213.83
Output dim: 1, lower bound: -79.3589439, upper bound: 79.2992963
Binary search (step 2): status=Status.VERIFIED, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=124.9945068359375
rel_dist={1: [-79.47797671394284, 79.47797670615168]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0234375
execution time: 11219.91 seconds
