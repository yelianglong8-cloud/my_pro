import pandas as pd
import numpy as np

# 1. 创建 Memristor (忆阻器) 电导数据
# 模拟论文中提到的约 30 个连续可调电导状态
memristor_states = np.linspace(0.1, 25.0, 30)
df_mem = pd.DataFrame(memristor_states)

# 保存为 Excel。由于原脚本指定了 header=None，因此我们不写入表头(header=False)
df_mem.to_excel('memristor conductance.xlsx', sheet_name='sheet_name', index=False, header=False)


# 2. 创建 GMC (类高斯记忆单元) 峰值电流数据
# 模拟论文中提到的 15 个由脉冲调制的峰值电流状态
gmc_states = np.linspace(1.5, 8.5, 15)
df_gmc = pd.DataFrame(gmc_states)

df_gmc.to_excel('GMC peak current.xlsx', sheet_name='sheet_name', index=False, header=False)

print("两个器件的数据表格已成功创建！")