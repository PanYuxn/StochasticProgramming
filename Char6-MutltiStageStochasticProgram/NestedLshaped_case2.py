import gurobipy as gp
from gurobipy import GRB
import numpy as np
from collections import defaultdict

class NestedLShapedAirConditioner:
    def __init__(self, tolerance=1e-6, max_iterations=100):
        """初始化嵌套L形法求解器
        
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
        self.optimality_cuts_stage3 = defaultdict(list)  # 第二阶段的最优性切割面，按情景索引
        self.feasibility_cuts_stage3 = defaultdict(list)  # 第二阶段的可行性切割面，按情景索引
        
        # 存储每一阶段的最优解
        self.opt_sol_stage1 = None
        self.opt_sol_stage2 = {}
        self.opt_sol_stage3 = {}
        
        # 初始化参数（所有单位除以100）
        self._init_params()
        
        # 创建环境和模型
        self.env = gp.Env(empty=True)
        self.env.setParam('OutputFlag', 0)  # 关闭Gurobi输出
        self.env.start()
    
    def _init_params(self):
        """初始化问题参数（所有单位除以100）"""
        # 成本参数
        self.regular_cost = 1.0     # 常规生产成本 $100/台 -> $1/台
        self.overtime_cost = 3.0    # 加班生产成本 $300/台 -> $3/台
        self.storage_cost = 0.5     # 库存成本 $50/台/月 -> $0.5/台/月
        
        # 生产能力
        self.regular_capacity = 2.0  # 常规生产能力 200台/月 -> 2.0台/月
        
        # 需求参数
        self.demand = {
            1: 1.0,                 # 第1月确定性需求: 100台 -> 1.0台
            (2, 'L'): 1.0,          # 第2月低需求: 100台 -> 1.0台
            (2, 'H'): 3.0,          # 第2月高需求: 300台 -> 3.0台
            (3, 'LL'): 1.0,         # 第3月低-低需求: 100台 -> 1.0台
            (3, 'LH'): 3.0,         # 第3月低-高需求: 300台 -> 3.0台
            (3, 'HL'): 1.0,         # 第3月高-低需求: 100台 -> 1.0台
            (3, 'HH'): 3.0          # 第3月高-高需求: 300台 -> 3.0台
        }
        
        # 概率
        self.prob = {
            (2, 'L'): 0.5,          # 第2月低需求概率
            (2, 'H'): 0.5,          # 第2月高需求概率
            (3, 'LL'): 0.5,        # 第3月低-低需求概率
            (3, 'LH'): 0.5,        # 第3月低-高需求概率
            (3, 'HL'): 0.5,        # 第3月高-低需求概率
            (3, 'HH'): 0.5         # 第3月高-高需求概率
        }
        
        # 情景名称映射
        self.stage2_scenarios = ['L', 'H']
        self.stage3_scenarios = {
            'L': ['LL', 'LH'],
            'H': ['HL', 'HH']
        }
    
    def build_first_stage_model(self):
        """构建第一阶段主问题模型"""
        model = gp.Model("First_Stage_Problem", env=self.env)
        
        # 决策变量
        x = model.addVar(lb=0.0, ub=self.regular_capacity, name="x_1")  # 常规生产
        w = model.addVar(lb=0.0, name="w_1")  # 加班生产
        y = model.addVar(lb=0.0, name="y_1")  # 库存
        theta = model.addVar(lb=0, name="theta_2")  # 后续期望成本
        
        # 目标函数: 第一阶段成本 + 后续期望成本
        model.setObjective(
            self.regular_cost * x + self.overtime_cost * w + self.storage_cost * y + theta, 
            GRB.MINIMIZE
        )
        
        # 生产量约束
        model.addConstr(x<=2 , name = 'x_constraint')
        
        # 需求平衡约束: 生产 = 需求 + 库存
        model.addConstr(x + w == self.demand[1] + y, name="balance_1")
        
        # 添加最优性切割面
        for i, (coef, rhs) in enumerate(self.optimality_cuts_stage2):
            model.addConstr(
                theta >= rhs - coef['x'] * x - coef['w'] * w - coef['y'] * y, 
                name=f"optimality_cut_{i}"
            )
        
        # 添加可行性切割面
        for i, (coef, rhs) in enumerate(self.feasibility_cuts_stage2):
            model.addConstr(
                coef['x'] * x + coef['w'] * w + coef['y'] * y <= rhs,
                name=f"feasibility_cut_{i}"
            )
        
        model.update()
        return model
    
    def build_second_stage_model(self, scenario, first_stage_sol):
        """构建第二阶段子问题模型
        
        Args:
            scenario: 第二阶段情景 ('L' 或 'H')
            first_stage_sol: 第一阶段解 (x_1, w_1, y_1)
            
        Returns:
            Gurobi模型
        """
        model = gp.Model(f"Second_Stage_Problem_{scenario}", env=self.env)
        
        # 提取第一阶段决策值
        x1_val, w1_val, y1_val = first_stage_sol
        
        # 决策变量
        x = model.addVar(lb=0.0, name=f"x_2_{scenario}")  # 常规生产
        w = model.addVar(lb=0.0, name=f"w_2_{scenario}")  # 加班生产
        y = model.addVar(lb=0.0, name=f"y_2_{scenario}")  # 库存
        theta = model.addVar(lb=0, name=f"theta_3_{scenario}")  # 后续期望成本
        
        # 目标函数: 第二阶段成本 + 后续期望成本
        model.setObjective(
            self.regular_cost * x + self.overtime_cost * w + self.storage_cost * y + theta, 
            GRB.MINIMIZE
        )
        
        # 生产量约束
        model.addConstr(x<=2 , name = f'x_constraint_{scenario}')
        
        # 需求平衡约束: 当前生产 + 前期库存 = 当前需求 + 结转库存
        model.addConstr(
            x + w + y1_val == self.demand[(2, scenario)] + y, 
            name=f"balance_2_{scenario}"
        )
        
        # 添加最优性切割面
        for i, (coef, rhs) in enumerate(self.optimality_cuts_stage3[scenario]):
            model.addConstr(
                theta >= rhs - coef['x'] * x - coef['w'] * w - coef['y'] * y, 
                name=f"optimality_cut_{scenario}_{i}"
            )
        
        # 添加可行性切割面
        for i, (coef, rhs) in enumerate(self.feasibility_cuts_stage3[scenario]):
            model.addConstr(
                coef['x'] * x + coef['w'] * w + coef['y'] * y <= rhs,
                name=f"feasibility_cut_{scenario}_{i}"
            )
        
        # # 为theta添加下限约束，确保边界情况下有合理值
        # # 使用该情景下最小可能成本作为下限
        # min_cost = min(self.regular_cost * min(self.regular_capacity, self.demand[(2, scenario)]), 
        #                self.overtime_cost * self.demand[(2, scenario)]) 
        # min_cost = min(min_cost for s in self.stage2_scenarios[scenario])
        # model.addConstr(theta >= min_cost, name=f"theta_lower_bound_{scenario}")
        
        model.update()
        return model
    
    def build_third_stage_model(self, scenario, second_stage_sol):
        """构建第三阶段子问题模型
        
        Args:
            scenario: 第三阶段情景 ('LL', 'LH', 'HL', 'HH')
            second_stage_sol: 第二阶段解 (x_2, w_2, y_2)
            
        Returns:
            Gurobi模型
        """
        model = gp.Model(f"Third_Stage_Problem_{scenario}", env=self.env)
        
        # 提取第二阶段决策值
        x2_val, w2_val, y2_val = second_stage_sol
        
        # 决策变量
        x = model.addVar(lb=0.0, name=f"x_3_{scenario}")  # 常规生产
        w = model.addVar(lb=0.0, name=f"w_3_{scenario}")  # 加班生产
        
        # 目标函数: 只有第三阶段成本
        model.setObjective(
            self.regular_cost * x + self.overtime_cost * w, 
            GRB.MINIMIZE
        )
        # 生产量约束
        model.addConstr(x<=2 , name = f'x_constraint_3_{scenario}')
        
        # 需求平衡约束: 当前生产 + 前期库存 = 当前需求
        model.addConstr(
            x + w + y2_val == self.demand[(3, scenario)], 
            name=f"balance_3_{scenario}"
        )
        
        model.update()
        return model
    
    def solve(self):
        """主求解程序"""
        print("开始求解空调生产多阶段随机规划问题...")
        
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
            x1_val = first_stage_model.getVarByName("x_1").x
            w1_val = first_stage_model.getVarByName("w_1").x
            y1_val = first_stage_model.getVarByName("y_1").x
            theta2_val = first_stage_model.getVarByName("theta_2").x
            first_stage_obj = first_stage_model.objVal - theta2_val  # 纯第一阶段成本
            self.LB = first_stage_model.objVal  # 当前下界
            
            print(f"  第一阶段解: x_1 = {x1_val:.4f}, w_1 = {w1_val:.4f}, y_1 = {y1_val:.4f}")
            print(f"  第一阶段成本: {first_stage_obj:.4f}, theta_2 = {theta2_val:.4f}")
            
            # 步骤2: 前向阶段 - 构建并求解第二和第三阶段子问题
            stage2_expected_cost = 0
            all_stage2_feasible = True
            all_stage3_feasible = True
            
            # 存储每个情景的信息，用于后向传递
            stage2_results = {}
            stage3_results = defaultdict(dict)
            
            for s2 in self.stage2_scenarios:
                # 构建并求解第二阶段子问题
                second_stage_model = self.build_second_stage_model(s2, (x1_val, w1_val, y1_val))
                second_stage_model.optimize()
                
                # 检查可行性
                if second_stage_model.status != GRB.OPTIMAL:
                    print(f"  第二阶段情景 {s2} 不可行!")
                    all_stage2_feasible = False

                # 获取第二阶段最优解
                x2_val = second_stage_model.getVarByName(f"x_2_{s2}").x
                w2_val = second_stage_model.getVarByName(f"w_2_{s2}").x
                y2_val = second_stage_model.getVarByName(f"y_2_{s2}").x
                theta3_val = second_stage_model.getVarByName(f"theta_3_{s2}").x
                balance_dual = second_stage_model.getConstrByName(f"balance_2_{s2}").pi
                product_dual = second_stage_model.getConstrByName(f"x_constraint_{s2}").pi
                # 计算第二阶段的直接成本
                stage2_direct_cost = self.regular_cost * x2_val + self.overtime_cost * w2_val + self.storage_cost * y2_val
                
                print(f"  第二阶段情景 {s2} 解: x_2 = {x2_val:.4f}, w_2 = {w2_val:.4f}, y_2 = {y2_val:.4f}, theta_3 = {theta3_val:.4f}")
                print(f"  第二阶段情景 {s2} 直接成本: {stage2_direct_cost:.4f}, 对偶值 = {balance_dual:.4f}, 生产量对偶值 = {product_dual:.4f}")
                
                self.opt_sol_stage2[s2] = (x2_val, w2_val, y2_val)
                
                # 构建并求解第三阶段子问题
                stage3_expected_cost = 0
                stage3_feasible = True
                
                for s3 in self.stage3_scenarios[s2]:
                    third_stage_model = self.build_third_stage_model(s3, (x2_val, w2_val, y2_val))
                    third_stage_model.optimize()
                    
                    # 检查可行性
                    if third_stage_model.status != GRB.OPTIMAL:
                        print(f"    第三阶段情景 {s3} 不可行!")
                        stage3_feasible = False
                        all_stage3_feasible = False
                        
                    # 获取第三阶段最优解和对偶值
                    x3_val = third_stage_model.getVarByName(f"x_3_{s3}").x
                    w3_val = third_stage_model.getVarByName(f"w_3_{s3}").x
                    balance_dual_s3 = third_stage_model.getConstrByName(f"balance_3_{s3}").pi
                    product_dual_s3 = third_stage_model.getConstrByName(f"x_constraint_3_{s3}").pi
                    # 计算第三阶段成本
                    stage3_cost = self.regular_cost * x3_val + self.overtime_cost * w3_val
                    
                    print(f"    第三阶段情景 {s3} 解: x_3 = {x3_val:.4f}, w_3 = {w3_val:.4f}, 成本 = {stage3_cost:.4f}, 对偶值 = {balance_dual_s3:.4f}, 生产量对偶值 = {product_dual_s3:.4f}")
                    
                    self.opt_sol_stage3[s3] = (x3_val, w3_val)
                    stage3_results[s2][s3] = {
                        'x3': x3_val,
                        'w3': w3_val,
                        'obj': stage3_cost,
                        'dual': balance_dual_s3,
                        'product_dual': product_dual_s3
                    }
                    
                    # 计算加权成本贡献
                    prob_s3 = self.prob.get((3, s3), 0)
                    stage3_expected_cost += prob_s3 * stage3_cost
                
                # 如果第三阶段所有情景都是可行的，记录第二阶段信息
                if stage3_feasible:
                    print(f"  第二阶段情景 {s2} 下第三阶段期望成本: {stage3_expected_cost:.4f}")
                    
                    # 存储第二阶段结果
                    stage2_results[s2] = {
                        'x2': x2_val,
                        'w2': w2_val,
                        'y2': y2_val,
                        'obj': stage2_direct_cost,
                        'dual': balance_dual,
                        'product_dual': product_dual,
                        'expected_s3_cost': stage3_expected_cost
                    }
                    
                    # 计算总加权成本贡献
                    prob_s2 = self.prob.get((2, s2), 0)
                    stage2_expected_cost += prob_s2 * (stage2_direct_cost + stage3_expected_cost)
                else:
                    print(f"  第二阶段情景 {s2} 下有不可行的第三阶段情景，跳过成本计算")
            
            # 如果有不可行情景，跳过其余步骤
            if not all_stage2_feasible or not all_stage3_feasible:
                print("  存在不可行的情景，添加可行性切割后继续")
                continue
            
            # 计算目标函数上界
            current_UB = first_stage_obj + stage2_expected_cost
            self.UB = min(self.UB, current_UB)
            print(f"  当前上界: {current_UB:.4f}, 最佳上界: {self.UB:.4f}")
            print(f"  当前下界: {self.LB:.4f}")
            
            # 步骤3: 后向阶段 - 生成切割面
            # 首先从第三阶段向第二阶段传递切割面
            for s2 in self.stage2_scenarios:
                if s2 not in stage2_results:
                    continue
                
                # 第三阶段最优性切割面
                # 这里使用公式计算，当前的矩阵T_k = [[0,0,0],[0,0,1]]
                T_k = np.array([[0,0,0],[0,0,1]])
                expected_dual = 0   
                expected_rhs = 0
                
                for s3 in self.stage3_scenarios[s2]:
                    if s3 in stage3_results[s2]:
                        dual = np.array([stage3_results[s2][s3]['product_dual'],stage3_results[s2][s3]['dual']])
                        e = np.array([2,self.demand[3,s3]])
                        
                        prob_s3 = self.prob.get((3, s3), 0)

                        expected_dual += prob_s3 * np.dot(dual, T_k)[-1]
                        expected_rhs += prob_s3 *  np.dot(dual, e)
                
                if expected_dual != 0:
                    # 创建切割面系数和常数项
                    coef = {'x': 0, 'w': 0, 'y': -expected_dual}
                    
                    # 添加切割面
                    self.optimality_cuts_stage3[s2].append((coef, expected_rhs))
                    print(f"  添加第三阶段最优性切割到第二阶段情景 {s2}: theta_3 >= {expected_rhs:.4f} + {expected_dual:.4f}*y_2")
            
            # 然后从第二阶段向第一阶段传递切割面
            if all_stage2_feasible:
                # 使用显式的切割面生成方法
                
                # 从对偶变量计算切割面系数和常数项
                expected_balance_dual = 0
                expected_product_dual = 0
                expected_rhs = 0
                
                for s2, results in stage2_results.items():
                    prob_s2 = self.prob.get((2, s2), 0)
                    balance_dual = results['dual']
                    product_dual = results['product_dual']
                    
                    expected_balance_dual += prob_s2 * balance_dual
                    expected_product_dual += prob_s2 * product_dual
                    expected_rhs += prob_s2 * (balance_dual * self.demand[(2, s2)] + product_dual * self.regular_capacity)
                # 检查是否有边界解（x2=0或y2=0）
                # 对于边界解，需要特殊处理
                has_boundary = False
                for s2, results in stage2_results.items():
                    if results['x2'] < 1e-6 or results['y2'] < 1e-6:
                        has_boundary = True
                        break
                
                if has_boundary or (abs(expected_balance_dual) < 1e-6 and abs(expected_product_dual) < 1e-6):
                    # 使用直接的成本函数计算切割面
                    # 计算总期望成本
                    total_expected_cost = stage2_expected_cost
                    
                    # 对当前解点进行估计
                    # y1增加1单位，可减少regular_cost单位的第二阶段成本
                    coef = {'x': 0, 'w': 0, 'y': -self.regular_cost}  
                    
                    # 计算截距
                    rhs = total_expected_cost + coef['y'] * y1_val
                    
                    # 添加切割面
                    self.optimality_cuts_stage2.append((coef, rhs))
                    print(f"  添加第二阶段最优性切割到第一阶段(直接法): theta_2 >= {rhs:.4f} + {-coef['y']:.4f}*y_1")
                else:
                    # 使用对偶变量生成切割面
                    coef = {
                        'x': 0, 
                        'w': 0, 
                        'y': -expected_balance_dual
                    }
                
                    # 添加切割面
                    self.optimality_cuts_stage2.append((coef, expected_rhs))
                    print(f"  添加第二阶段最优性切割到第一阶段: theta_2 >= {expected_rhs:.4f} + {expected_dual:.4f}*y_1")
                
                # 验证切割面
                if has_boundary or abs(expected_dual) < 1e-6:
                    expected_theta2 = rhs + coef['y'] * y1_val
                else:
                    expected_theta2 = expected_rhs + coef['y'] * y1_val
                
                print(f"  切割面验证: 估计theta2 = {expected_theta2:.4f}, 实际期望成本 = {stage2_expected_cost:.4f}")
            
            # 检查收敛
            gap = abs(self.UB - self.LB) / (1e-10 + abs(self.UB))
            print(f"  相对间隙: {gap:.6f}")
            
            if gap < self.tolerance:
                print(f"\n算法在第 {self.iteration} 次迭代后收敛!")
                self.opt_sol_stage1 = (x1_val, w1_val, y1_val)
                break
        
        if self.iteration >= self.max_iterations:
            print(f"\n算法达到最大迭代次数 {self.max_iterations}，但未收敛")
            print(f"  最终相对间隙: {gap:.6f}")
            self.opt_sol_stage1 = (x1_val, w1_val, y1_val)
        
        # 打印最终结果
        self._print_results()
        
        return {
            'stage1': self.opt_sol_stage1,
            'stage2': self.opt_sol_stage2,
            'stage3': self.opt_sol_stage3,
            'obj': self.UB
        }
    
    def _print_results(self):
        """打印最终求解结果"""
        if self.opt_sol_stage1 is None:
            print("没有找到可行解")
            return
            
        x1, w1, y1 = self.opt_sol_stage1
        
        print("\n====== 最终求解结果 ======")
        print(f"最优目标值: {self.UB:.4f} (单位: $100)")
        print(f"\n第一阶段决策:")
        print(f"  常规生产量(x_1): {x1:.4f} (单位: 100台)")
        print(f"  加班生产量(w_1): {w1:.4f} (单位: 100台)")
        print(f"  库存量(y_1): {y1:.4f} (单位: 100台)")
        
        print("\n第二阶段决策:")
        for s2, (x2, w2, y2) in self.opt_sol_stage2.items():
            print(f"  情景 {s2} (概率 = {self.prob.get((2, s2), 0)}):")
            print(f"    常规生产量(x_2): {x2:.4f} (单位: 100台)")
            print(f"    加班生产量(w_2): {w2:.4f} (单位: 100台)")
            print(f"    库存量(y_2): {y2:.4f} (单位: 100台)")
        
        print("\n第三阶段决策:")
        for s3, (x3, w3) in self.opt_sol_stage3.items():
            print(f"  情景 {s3} (概率 = {self.prob.get((3, s3), 0)}):")
            print(f"    常规生产量(x_3): {x3:.4f} (单位: 100台)")
            print(f"    加班生产量(w_3): {w3:.4f} (单位: 100台)")
        
        print("\n优化策略解读:")
        print(f"  第一月: 常规生产 {x1:.2f} 单位，加班生产 {w1:.2f} 单位，存储 {y1:.2f} 单位")
        for s2, (x2, w2, y2) in self.opt_sol_stage2.items():
            print(f"  第二月 (情景 {s2}): 常规生产 {x2:.2f} 单位，加班生产 {w2:.2f} 单位，存储 {y2:.2f} 单位")
        
        cost_breakdown = {
            'stage1_regular': self.regular_cost * x1,
            'stage1_overtime': self.overtime_cost * w1,
            'stage1_storage': self.storage_cost * y1,
            'stage2': {},
            'stage3': {}
        }
        
        for s2, (x2, w2, y2) in self.opt_sol_stage2.items():
            cost_breakdown['stage2'][s2] = {
                'regular': self.regular_cost * x2,
                'overtime': self.overtime_cost * w2,
                'storage': self.storage_cost * y2
            }
        
        for s3, (x3, w3) in self.opt_sol_stage3.items():
            cost_breakdown['stage3'][s3] = {
                'regular': self.regular_cost * x3,
                'overtime': self.overtime_cost * w3
            }
        
        print("\n成本明细:")
        print(f"  第一阶段:")
        print(f"    常规生产成本: {cost_breakdown['stage1_regular']:.4f}")
        print(f"    加班生产成本: {cost_breakdown['stage1_overtime']:.4f}")
        print(f"    库存成本: {cost_breakdown['stage1_storage']:.4f}")
        
        stage1_total = cost_breakdown['stage1_regular'] + cost_breakdown['stage1_overtime'] + cost_breakdown['stage1_storage']
        print(f"    第一阶段合计: {stage1_total:.4f}")
        
        print("\n  第二阶段期望成本:")
        stage2_total = 0
        for s2 in self.stage2_scenarios:
            if s2 in cost_breakdown['stage2']:
                s2_costs = cost_breakdown['stage2'][s2]
                s2_total = s2_costs['regular'] + s2_costs['overtime'] + s2_costs['storage']
                prob_s2 = self.prob.get((2, s2), 0)
                stage2_total += prob_s2 * s2_total
                print(f"    情景 {s2} (概率 = {prob_s2}):")
                print(f"      常规生产成本: {s2_costs['regular']:.4f}")
                print(f"      加班生产成本: {s2_costs['overtime']:.4f}")
                print(f"      库存成本: {s2_costs['storage']:.4f}")
                print(f"      合计: {s2_total:.4f}")
        
        print(f"    第二阶段期望合计: {stage2_total:.4f}")
        
        print("\n  第三阶段期望成本:")
        stage3_total = 0
        for s3 in self.opt_sol_stage3.keys():
            if s3 in cost_breakdown['stage3']:
                s3_costs = cost_breakdown['stage3'][s3]
                s3_total = s3_costs['regular'] + s3_costs['overtime']
                prob_s3 = self.prob.get((3, s3), 0)
                stage3_total += prob_s3 * s3_total
                print(f"    情景 {s3} (概率 = {prob_s3}):")
                print(f"      常规生产成本: {s3_costs['regular']:.4f}")
                print(f"      加班生产成本: {s3_costs['overtime']:.4f}")
                print(f"      合计: {s3_total:.4f}")
        
        print(f"    第三阶段期望合计: {stage3_total:.4f}")
        print(f"\n  总期望成本: {stage1_total + stage2_total + stage3_total:.4f}")
        
    def __del__(self):
        """析构函数，释放Gurobi环境资源"""
        if hasattr(self, 'env'):
            self.env.close()


# 运行主程序
if __name__ == "__main__":
    solver = NestedLShapedAirConditioner(tolerance=1e-4, max_iterations=20)
    results = solver.solve()