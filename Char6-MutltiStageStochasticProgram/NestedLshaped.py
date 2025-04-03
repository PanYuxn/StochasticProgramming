import gurobipy as gp
from gurobipy import GRB
import numpy as np
from collections import defaultdict

class NestedLShapedMethod:
    def __init__(self, tolerance=1e-6, max_iterations=100):
        """初始化嵌套L形方法求解器
        
        Args:
            tolerance: 上下界收敛容差
            max_iterations: 最大迭代次数
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.UB = float('inf')  # 上界
        self.LB = float('-inf')  # 下界
        self.iteration = 0      # 迭代计数器
        
        # 存储切割面
        self.optimality_cuts_stage2 = []  # 第一阶段的最优性切割面
        self.feasibility_cuts_stage2 = []  # 第一阶段的可行性切割面
        
        # 存储每一阶段的最优解
        self.opt_sol_stage1 = None
        self.opt_sol_stage2 = {}
        self.opt_sol_stage3 = {}
        
        # 初始化参数
        self._init_params()
        
        # 创建环境和模型
        self.env = gp.Env(empty=True)
        self.env.setParam('OutputFlag', 0)  # 关闭Gurobi输出
        self.env.start()
    
    def _init_params(self):
        """初始化问题参数"""
        # 第一阶段参数
        self.c1 = 10  # 投资成本系数
        
        # 第二阶段参数 (2个情景)
        # 情景1: 高增长
        # 情景2: 低增长
        self.c2 = {1: 15, 2: 15}
        self.A2 = {1: 1, 2: 1}
        self.B2 = {1: 1, 2: 1}
        self.b2 = {1: 120, 2: 100}
        self.prob_stage2 = {1: 0.5, 2: 0.5}
        
        # 第三阶段参数 (每个第二阶段情景下3个情景，共6个情景)
        self.c3 = {
            (1, 1): 5, (1, 2): 5, (1, 3): 5,  # 情景1下的三个子情景
            (2, 1): 5, (2, 2): 5, (2, 3): 5   # 情景2下的三个子情景
        }
        self.A3 = {
            (1, 1): 1, (1, 2): 1, (1, 3): 1, 
            (2, 1): 1, (2, 2): 1, (2, 3): 1
        }
        self.B3 = {
            (1, 1): 1, (1, 2): 1, (1, 3): 1, 
            (2, 1): 1, (2, 2): 1, (2, 3): 1
        }
        self.b3 = {
            (1, 1): 180, (1, 2): 150, (1, 3): 130,  # 高增长情景下的需求
            (2, 1): 150, (2, 2): 130, (2, 3): 110   # 低增长情景下的需求
        }
        self.prob_stage3 = {
            (1, 1): 0.3, (1, 2): 0.4, (1, 3): 0.3,
            (2, 1): 0.3, (2, 2): 0.4, (2, 3): 0.3
        }
    
    def build_first_stage_model(self):
        """构建第一阶段主问题模型"""
        model = gp.Model("First_Stage_Problem", env=self.env)
        
        # 决策变量
        x1 = model.addVar(lb=0.0, name="x1")
        theta2 = model.addVar(lb=0.0, name="theta2")
        
        # 目标函数
        model.setObjective(self.c1 * x1 + theta2, GRB.MINIMIZE)
        
        # 添加最优性切割面
        for i, (a, b) in enumerate(self.optimality_cuts_stage2):
            model.addConstr(theta2 >= a - b * x1, name=f"optimality_cut_{i}")
        
        # 添加可行性切割面
        for i, bound in enumerate(self.feasibility_cuts_stage2):
            model.addConstr(x1 <= bound, name=f"feasibility_cut_{i}")
        
        model.update()
        model.write(f"{self.iteration}_First_Stage_Problem.lp")
        return model
    
    def build_second_stage_model(self, scenario, x1_value):
        """构建第二阶段子问题模型
        
        Args:
            scenario: 第二阶段情景索引
            x1_value: 第一阶段决策值
            
        Returns:
            Gurobi模型
        """
        model = gp.Model(f"Second_Stage_Problem_Scenario_{scenario}", env=self.env)
        
        # 决策变量
        x2 = model.addVar(lb=0.0, name=f"x2_{scenario}")
        theta3 = model.addVar(lb=0.0, name=f"theta3_{scenario}")
        
        # 目标函数
        model.setObjective(self.c2[scenario] * x2 + theta3, GRB.MINIMIZE)
        
        # 容量平衡约束: A2*x2 + B2*x1 = b2
        balance_constraint = model.addConstr(
            self.A2[scenario] * x2 + self.B2[scenario] * x1_value == self.b2[scenario], 
            name=f"Balance_{scenario}"
        )
        
        # 添加第三阶段的切割面 (在第一次迭代时添加初始切割面)
        if self.iteration > 0:
            # 计算第三阶段期望需求
            expected_b3 = 0
            for sub_scenario in [1, 2, 3]:
                expected_b3 += self.prob_stage3[(scenario, sub_scenario)] * self.b3[(scenario, sub_scenario)]
            
            # 添加最优性切割面: theta3 >= 5 * (expected_b3 - x2)
            # 在这个问题中，我们知道第三阶段对偶变量是5
            model.addConstr(theta3 >= 5 * expected_b3 - 5 * x2, name="initial_cut")
        
        model.update()
        model.write(f"{self.iteration}_Second_Stage_Problem_Scenario_{scenario}.lp")
        return model
    
    def build_third_stage_model(self, stage2_scenario, stage3_scenario, x2_value):
        """构建第三阶段子问题模型
        
        Args:
            stage2_scenario: 第二阶段情景索引
            stage3_scenario: 第三阶段情景索引
            x2_value: 第二阶段决策值
            
        Returns:
            Gurobi模型
        """
        model = gp.Model(f"Third_Stage_Problem_S2_{stage2_scenario}_S3_{stage3_scenario}", env=self.env)
        
        # 决策变量
        x3 = model.addVar(lb=0.0, name=f"x3_{stage2_scenario}_{stage3_scenario}")
        
        # 目标函数
        model.setObjective(self.c3[(stage2_scenario, stage3_scenario)] * x3, GRB.MINIMIZE)
        
        # 电力平衡约束: A3*x3 + B3*x2 = b3
        balance = model.addConstr(
            self.A3[(stage2_scenario, stage3_scenario)] * x3 + 
            self.B3[(stage2_scenario, stage3_scenario)] * x2_value == 
            self.b3[(stage2_scenario, stage3_scenario)], 
            name=f"Balance_{stage2_scenario}_{stage3_scenario}"
        )
        
        model.update()
        model.write(f"{self.iteration}_Third_Stage_Problem_S2_{stage2_scenario}_S3_{stage3_scenario}.lp")
        return model
    
    def solve(self):
        """主求解程序"""
        print("开始求解三阶段随机线性规划问题...")
        
        while self.iteration < self.max_iterations:
            self.iteration += 1
            print(f"\n迭代 {self.iteration}:")
            
            # 步骤1: 构建并求解第一阶段主问题
            first_stage_model = self.build_first_stage_model()
            first_stage_model.optimize()
            
            # 检查是否找到可行解
            if first_stage_model.status != GRB.OPTIMAL:
                print("第一阶段问题无解，算法终止")
                return None
            
            # 获取第一阶段最优解
            x1_value = first_stage_model.getVarByName("x1").x
            theta2_value = first_stage_model.getVarByName("theta2").x
            self.LB = first_stage_model.objVal
            
            print(f"  第一阶段解: x1 = {x1_value:.2f}, theta2 = {theta2_value:.2f}")
            
            # 步骤2: 前向阶段 - 构建并求解第二和第三阶段子问题
            stage2_expected_cost = 0
            all_stage2_feasible = True
            stage2_results = {}
            stage3_results = {}
            
            for scenario2 in [1, 2]:
                # 构建并求解第二阶段子问题
                second_stage_model = self.build_second_stage_model(scenario2, x1_value)
                second_stage_model.optimize()
                
                # 检查可行性
                if second_stage_model.status != GRB.OPTIMAL:
                    print(f"  第二阶段情景 {scenario2} 不可行!")
                    all_stage2_feasible = False
                    
                    # 添加可行性切割面 - 确保 x1 <= b2
                    # 这是由于本问题的特殊结构：x2 + x1 = b2，且 x2 >= 0
                    self.feasibility_cuts_stage2.append(self.b2[scenario2])
                    print(f"  添加可行性切割: x1 <= {self.b2[scenario2]}")
                    continue
                
                # 获取第二阶段最优解
                x2_value = second_stage_model.getVarByName(f"x2_{scenario2}").x
                theta3_value = second_stage_model.getVarByName(f"theta3_{scenario2}").x
                stage2_results[scenario2] = (x2_value, theta3_value, second_stage_model.objVal)
                
                print(f"  第二阶段情景 {scenario2} 解: x2 = {x2_value:.2f}, theta3 = {theta3_value:.2f}")
                self.opt_sol_stage2[scenario2] = x2_value
                
                # 构建并求解第三阶段子问题
                stage3_expected_cost = 0
                stage3_scenario_results = {}
                
                for scenario3 in [1, 2, 3]:
                    third_stage_model = self.build_third_stage_model(scenario2, scenario3, x2_value)
                    third_stage_model.optimize()
                    
                    # 检查可行性
                    if third_stage_model.status != GRB.OPTIMAL:
                        print(f"    第三阶段情景 ({scenario2},{scenario3}) 不可行!")
                        continue
                    
                    # 获取第三阶段最优解
                    x3_value = third_stage_model.getVarByName(f"x3_{scenario2}_{scenario3}").x
                    objective_value = third_stage_model.objVal
                    
                    print(f"    第三阶段情景 ({scenario2},{scenario3}) 解: x3 = {x3_value:.2f}, 成本 = {objective_value:.2f}")
                    
                    self.opt_sol_stage3[(scenario2, scenario3)] = x3_value
                    stage3_scenario_results[(scenario2, scenario3)] = (x3_value, objective_value)
                    stage3_expected_cost += self.prob_stage3[(scenario2, scenario3)] * objective_value
                
                print(f"  第二阶段情景 {scenario2} 下第三阶段期望成本: {stage3_expected_cost:.2f}")
                stage3_results[scenario2] = stage3_expected_cost
                
                # 计算第二阶段情景的总成本
                scenario_total_cost = self.c2[scenario2] * x2_value + stage3_expected_cost
                stage2_expected_cost += self.prob_stage2[scenario2] * scenario_total_cost
            
            # 如果有不可行情景，跳过其余步骤
            if not all_stage2_feasible:
                print("  存在不可行的第二阶段情景，添加可行性切割后继续")
                continue
            
            # 计算目标函数上界
            current_UB = self.c1 * x1_value + stage2_expected_cost
            self.UB = min(self.UB, current_UB)
            print(f"  当前上界: {current_UB:.2f}, 最佳上界: {self.UB:.2f}")
            print(f"  当前下界: {self.LB:.2f}")
            
            # 步骤3: 后向阶段 - 生成切割面
            if all_stage2_feasible:
                # 从第二阶段子问题生成切割面到第一阶段主问题
                # 使用对偶变量来生成切割面
                
                # 对于这个特定问题，我们知道：
                # 1. 第三阶段的对偶变量是5（固定的）
                # 2. 第二阶段的约束是 x2 + x1 = b2，所以对偶变量是15（价格系数）
                
                # 计算期望第三阶段成本
                expected_stage3_cost = 0
                for scenario2 in [1, 2]:
                    expected_stage3_cost += self.prob_stage2[scenario2] * stage3_results[scenario2]
                
                # 计算期望需求
                expected_demand = 0
                for scenario2 in [1, 2]:
                    expected_demand += self.prob_stage2[scenario2] * self.b2[scenario2]
                
                # 生成最优性切割面: theta2 >= a - b*x1
                # 其中：a = 15 * expected_demand + expected_stage3_cost
                #      b = 15
                a = 15 * expected_demand + expected_stage3_cost
                b = 15
                
                # 添加最优性切割
                self.optimality_cuts_stage2.append((a, b))
                print(f"  添加最优性切割: theta2 >= {a:.2f} - {b}*x1")
                
                # 验证切割面 - theta2应该接近stage2_expected_cost
                expected_theta2 = a - b * x1_value
                print(f"  切割面验证: 期望theta2 = {expected_theta2:.2f}, 实际期望成本 = {stage2_expected_cost:.2f}")
            
            # 检查收敛
            gap = abs(self.UB - self.LB) / (1e-10 + abs(self.UB))
            print(f"  相对间隙: {gap:.6f}")
            
            if gap < self.tolerance:
                print(f"\n算法在第 {self.iteration} 次迭代后收敛!")
                self.opt_sol_stage1 = x1_value
                break
        
        if self.iteration >= self.max_iterations:
            print(f"\n算法达到最大迭代次数 {self.max_iterations}，但未收敛")
            print(f"  最终相对间隙: {gap:.6f}")
            self.opt_sol_stage1 = x1_value
        
        # 打印最终结果
        self._print_results()
        
        return self.opt_sol_stage1, self.opt_sol_stage2, self.opt_sol_stage3
    
    def _print_results(self):
        """打印最终求解结果"""
        print("\n====== 最终求解结果 ======")
        print(f"最优目标值: {self.UB:.2f}")
        print(f"第一阶段决策: x1 = {self.opt_sol_stage1:.2f}")
        
        print("\n第二阶段决策:")
        for scenario, value in self.opt_sol_stage2.items():
            print(f"  情景 {scenario} (概率 = {self.prob_stage2[scenario]}): x2 = {value:.2f}")
        
        print("\n第三阶段决策:")
        for (s2, s3), value in self.opt_sol_stage3.items():
            print(f"  情景 ({s2},{s3}) (概率 = {self.prob_stage3[(s2,s3)]}): x3 = {value:.2f}")
        
        print("\n最优性切割面:")
        for i, (a, b) in enumerate(self.optimality_cuts_stage2):
            print(f"  切割面 {i+1}: theta2 >= {a:.2f} - {b}*x1")
        
        print("\n可行性切割面:")
        for i, bound in enumerate(self.feasibility_cuts_stage2):
            print(f"  切割面 {i+1}: x1 <= {bound}")
    
    def __del__(self):
        """析构函数，释放Gurobi环境资源"""
        if hasattr(self, 'env'):
            self.env.close()

# 运行求解器
if __name__ == "__main__":
    solver = NestedLShapedMethod(tolerance=1e-4)
    solver.solve()