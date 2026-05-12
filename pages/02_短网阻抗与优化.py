import streamlit as st
import numpy as np
import pandas as pd
import math

# 设置页面配置 (多页面模式下会自动继承主页配置，这里为了独立运行也加上)
st.set_page_config(page_title="短网阻抗优化", page_icon="⚡", layout="wide")

st.title("⚡ 矿热炉短网阻抗与功率因数优化")
st.markdown("本模块集成于**杨波的智能钢包设计系统**，用于快速核算短网参数（截面积、间距）对电炉功率因数及电阻热的影响。")

st.divider()

# ----------------- 输入参数区 -----------------
st.subheader("⚙️ 物理与几何参数输入")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**电气与材质参数**")
    f = st.number_input("电源频率 (Hz)", value=50.0, step=1.0)
    # 20°C 铜电阻率约 1.72e-8，运行高温下通常取 2.2e-8 左右
    rho_temp = st.number_input("运行温度下铜电阻率 (Ω·m)", value=2.2e-8, format="%.2e")
    L = st.number_input("短网等效长度 $L$ (m)", value=10.0, step=0.5)

with col2:
    st.markdown("**铜管截面参数**")
    D_outer = st.number_input("单根铜管外径 (mm)", value=60.0, step=1.0)
    t = st.number_input("铜管壁厚 (mm)", value=12.5, step=0.5)
    n_tubes = st.number_input("单相并联铜管数", value=4, step=1, min_value=1)

with col3:
    st.markdown("**系统工况参数**")
    D_phase = st.number_input("设计相间距 / 中心距 $D$ (mm)", value=100.0, step=5.0)
    I_total = st.number_input("单相总电流 (kA)", value=50.0, step=5.0)

# ----------------- 核心计算逻辑 -----------------
def calculate_impedance(D_phase_m):
    """根据间距计算短网的阻抗参数"""
    mu_0 = 4 * math.pi * 1e-7  # 真空磁导率
    
    # 1. 集肤深度计算 (Skin Depth)
    delta = math.sqrt(rho_temp / (math.pi * f * mu_0))
    
    # 2. 有效导电截面积计算
    r_out = (D_outer / 2) / 1000
    r_in = r_out - (t / 1000)
    
    # 如果集肤深度小于壁厚，则只取集肤深度作为有效导电层
    eff_thickness = min(delta, t / 1000)
    r_inner_eff = r_out - eff_thickness
    
    A_single = math.pi * (r_out**2 - r_inner_eff**2)
    A_total = A_single * n_tubes
    
    # 3. 交流电阻 R
    # 注：此处暂未加入邻近效应惩罚系数，作为基础工程核算
    R = rho_temp * L / A_total
    
    # 4. 感抗 X
    # 计算等效半径 (简化模型)
    r_eq = r_out * math.sqrt(n_tubes)
    
    # 防止极端输入导致数学域错误
    safe_D = max(D_phase_m, r_eq * 1.05)
    
    L_s = (mu_0 * L / math.pi) * math.log(safe_D / r_eq)
    X = 2 * math.pi * f * L_s
    
    # 5. 总阻抗与功率因数
    Z = math.sqrt(R**2 + X**2)
    PF = R / Z
    
    # 6. 电阻热损耗 (I^2 * R) kW
    P_loss = ((I_total * 1000)**2 * R) / 1000
    
    return R, X, Z, PF, delta, A_total, P_loss

# 获取当前设计参数的计算结果
R, X, Z, PF, delta, A_total, P_loss = calculate_impedance(D_phase / 1000)

# ----------------- 结果展示区 -----------------
st.subheader("📊 运行核算结果")

# 核心指标卡片
m1, m2, m3, m4 = st.columns(4)
m1.metric("单相有效导电截面积", f"{A_total * 1e6:,.1f} mm²", help="已考虑集肤效应折算")
m2.metric("当前功率因数 (PF)", f"{PF:.4f}")
m3.metric("单相电阻热损耗", f"{P_loss:,.1f} kW", help="仅为电阻发热，不包含感应损耗")
m4.metric("铜管内水流速 (估算)", f"{(I_total*1000 / (A_total*1e6)) * 0.05:.2f} m/s", help="简化占位，需结合实际水泵扬程")

st.markdown(f"**诊断信息：** 在 {f} Hz 下，当前设定温度的集肤深度理论值为 **{delta * 1000:.2f} mm**。")

st.divider()

# ----------------- 优化趋势图表 -----------------
st.subheader("📈 短网间距优化分析")
st.markdown("通过调整相邻导体间距，观察电磁补偿效应对**功率因数**和**无功损耗**的影响。")

# 生成间距从 50mm 到 200mm 的变化数据
distance_range = np.linspace(max(D_outer, 50), max(D_phase * 2, 200), 50)
data = []

for d in distance_range:
    res = calculate_impedance(d / 1000)
    data.append({
        "相间距 (mm)": d,
        "功率因数 (PF)": res[3],
        "感抗 X (mΩ)": res[1] * 1000
    })

df = pd.DataFrame(data).set_index("相间距 (mm)")

# 使用 Streamlit 原生双轴图表
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("**功率因数随间距变化曲线**")
    st.line_chart(df[["功率因数 (PF)"]], color="#FF4B4B")

with col_chart2:
    st.markdown("**感抗随间距变化曲线**")
    st.line_chart(df[["感抗 X (mΩ)"]], color="#0068C9")

st.info("💡 **工程提示：** 曲线证明了缩小间距可有效降低感抗并提升功率因数。但在实际设计中，当间距极度缩小时，需警惕**邻近效应**导致的交流电阻急剧攀升以及局部过热风险，应选取曲线中的平滑拐点作为设计‘甜点’。")
