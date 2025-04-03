\ Model Second_Stage_Problem_Scenario_2
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  15 x2_2 + theta3_2
Subject To
 Balance_2: x2_2 = 0
 initial_cut: 5 x2_2 + theta3_2 >= 650
Bounds
End
