import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from math import pi, sqrt, ceil, floor, cos, radians
import os

# ==========================================
# 0. 全局配置与工具函数
# ==========================================
st.set_page_config(
    page_title="机械设计专家系统 Pro",
    layout="wide",
    page_icon="⚙️",
    initial_sidebar_state="expanded"
)

# --- 样式优化 ---
st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: bold; color: #1E3A8A; }
    .metric-card { background-color: #F0F2F6; padding: 15px; border-radius: 10px; border-left: 5px solid #1E3A8A; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #F0F2F6; border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #FFFFFF; border-bottom: 2px solid #1E3A8A; }
</style>
""", unsafe_allow_html=True)

# --- 字体加载 ---
@st.cache_resource
def configure_fonts():
    # 尝试加载中文字体，按优先级
    font_candidates = ["SimHei.ttf", "simhei.ttf", "msyh.ttc", "simsun.ttc"]
    found_font = None
    for f in font_candidates:
        if os.path.exists(f):
            found_font = f
            break
    
    if found_font:
        fm.fontManager.addfont(found_font)
        prop = fm.FontProperties(fname=found_font)
        return prop.get_name(), True
    else:
        # Linux/Cloud 环境备选
        return ["WenQuanYi Micro Hei", "sans-serif"], False

font_family, is_font_success = configure_fonts()
plt.rcParams['font.sans-serif'] = [font_family] if isinstance(font_family, str) else font_family
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 核心数据库 (模拟第1卷、第2卷数据)
# ==========================================

# 材料库 (第1卷)
MATERIAL_DB = {
    "45钢 (调质)": {"sigma_b": 600, "sigma_s": 355, "HB": 240, "A0": 118, "E": 206000},
    "40Cr (调质)": {"sigma_b": 785, "sigma_s": 540, "HB": 260, "A0": 110, "E": 211000},
    "35SiMn (调质)": {"sigma_b": 885, "sigma_s": 735, "HB": 270, "A0": 105, "E": 210000},
    "Q235-A": {"sigma_b": 370, "sigma_s": 235, "HB": 140, "A0": 130, "E": 200000},
    "20CrMnTi (渗碳淬火)": {"sigma_b": 1080, "sigma_s": 835, "HB": 600, "A0": 100, "E": 212000},
    "自定义材料": {"sigma_b": 500, "sigma_s": 300, "HB": 200, "A0": 120, "E": 206000}
}

# 矿热炉经验系数 (用户Excel提取)
FURNACE_DB = {
    "硅锰 (SiMn)":     {"Ke": 6.3,  "J": 5.5, "Ky": 2.7,  "Ki": 6.4,  "Kh": 2.5},
    "高碳铬铁 (FeCr)": {"Ke": 6.8,  "J": 5.7, "Ky": 2.65, "Ki": 6.3,  "Kh": 2.6},
    "镍铁 (FeNi-RKEF)":{"Ke": 12.0, "J": 4.0, "Ky": 3.6,  "Ki": 10.0, "Kh": 2.9},
    "硅铁75 (FeSi75)": {"Ke": 6.8,  "J": 6.5, "Ky": 2.25, "Ki": 5.8,  "Kh": 2.2},
    "电石 (CaC2)":     {"Ke": 6.5,  "J": 7.0, "Ky": 2.7,  "Ki": 6.4,  "Kh": 2.2},
    "工业硅 (Si)":     {"Ke": 7.5,  "J": 6.0, "Ky": 2.4,  "Ki": 6.0,  "Kh": 2.3},
    "自定义":          {"Ke": 6.5,  "J": 5.5, "Ky": 2.7,  "Ki": 6.5,  "Kh": 2.5}
}

# 普通螺纹标准 (第2卷)
THREAD_DB = pd.DataFrame({
    "d": [6, 8, 10, 12, 16, 20, 24, 30, 36, 42, 48],
    "P": [1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5],
    "d2": [5.350, 7.188, 9.026, 10.863, 14.701, 18.376, 22.051, 27.727, 33.402, 39.077, 44.752],
    "As": [20.1, 36.6, 58.0, 84.3, 157, 245, 353, 561, 817, 1120, 1470] # 应力截面积
})

# ==========================================
# 2. 核心算法逻辑
# ==========================================

# --- A. 键槽推荐 (GB/T 1096) ---
def recommend_key(d):
    # 简化版查表
    if d <= 12: return 4, 4
    if d <= 17: return 5, 5
    if d <= 22: return 6, 6
    if d <= 30: return 8, 7
    if d <= 38: return 10, 8
    if d <= 44: return 12, 8
    if d <= 50: return 14, 9
    if d <= 58: return 16, 10
    if d <= 65: return 18, 11
    if d <= 75: return 20, 12
    if d <= 85: return 22, 14
    return 25, 14

# --- B. 齿轮接触强度简易计算 (Vol 3) ---
def calc_gear_sigma_H(T1, u, a, b, K=1.2, Zh=2.5, Ze=189.8):
    # T1: N.mm, u: 传动比, a: 中心距 mm, b: 齿宽 mm
    # 接触应力公式 sigma_H = Ze * sqrt( (2*K*T1*(u+1)) / (b * d1^2 * u) ) 
    # 此处使用中心距公式反推: d1 = 2a / (u+1)
    d1 = 2 * a / (u + 1)
    if d1 <= 0 or b <= 0: return 0
    sigma_H = Ze * sqrt((2 * K * T1 * (u + 1)) / (b * (d1**2) * u))
    return sigma_H

# ==========================================
# 3. 界面逻辑：主导航
# ==========================================

# 初始化 Session State
if 'current_module' not in st.session_state:
    st.session_state.current_module = "🔥 矿热电炉设计"

with st.sidebar:
    st.title("⚙️ 导航中心")
    st.markdown("基于《机械设计手册》V6")
    
    selected_module = st.radio(
        "选择功能模块:",
        ["🔥 矿热电炉设计", "🔩 轴系设计 (Vol.2)", "⚙️ 齿轮传动 (Vol.3)", "🔗 连接紧固 (Vol.2)", "📚 综合查询"]
    )
    
    st.info("💡 提示：所有计算结果均可导出CSV报表。")

# ==========================================
# 模块 1: 矿热电炉设计 (您的核心需求)
# ==========================================
if selected_module == "🔥 矿热电炉设计":
    st.header("🔥 矿热电炉全参数设计平台")
    st.markdown("集成 **容量计算、几何设计、导电系统配置、工程圆整** 四位一体。")

    # --- 1.1 数据状态管理 (圆整值存储) ---
    if 'furnace_recalc' not in st.session_state:
        st.session_state.furnace_recalc = True
    
    def trigger_furnace_recalc():
        st.session_state.furnace_recalc = True

    # --- 1.2 输入界面 ---
    col_l, col_r = st.columns([1, 1.5])
    
    with col_l:
        st.subheader("1. 基础工况输入")
        with st.expander("🛠️ 核心参数设定", expanded=True):
            alloy_type = st.selectbox("冶炼品种", list(FURNACE_DB.keys()), on_change=trigger_furnace_recalc)
            cap_mva = st.number_input("变压器容量 (MVA)", value=33.0, step=0.5, on_change=trigger_furnace_recalc)
            u1_kv = st.selectbox("一次侧电压 (kV)", [110, 35, 10, 220], index=1, on_change=trigger_furnace_recalc)
            lining_thick = st.number_input("平均炉衬厚度 (mm)", value=1200, step=100, on_change=trigger_furnace_recalc)

        with st.expander("🔧 导电系统配置 (铜瓦/铜管)", expanded=True):
            tile_num = st.number_input("单相铜瓦数量", 4, 16, 8, help="电极把持器铜瓦数")
            c_t1, c_t2 = st.columns(2)
            tube_d = c_t1.selectbox("铜管外径 Φ", [50,60,70,80,90,100], index=2)
            tube_t = c_t2.selectbox("铜管壁厚", [10,12.5,15,20], index=1)
            # 自动逻辑：铜管数量 = 2 * 铜瓦数量
            tube_num = tile_num * 2
            st.caption(f"ℹ️ 自动计算：单相铜管数量 = **{tube_num}** 根")

        with st.expander("🎛️ 经验系数微调 (Expert Mode)"):
            defaults = FURNACE_DB[alloy_type]
            ke = st.slider("电压系数 Ke", 1.0, 15.0, defaults['Ke'], 0.1, on_change=trigger_furnace_recalc)
            j_den = st.slider("电流密度 J", 1.0, 10.0, defaults['J'], 0.1, on_change=trigger_furnace_recalc)
            ky = st.number_input("极心圆系数 Ky", value=defaults['Ky'], step=0.05, on_change=trigger_furnace_recalc)
            ki = st.number_input("炉膛内径系数 Ki", value=defaults['Ki'], step=0.1, on_change=trigger_furnace_recalc)
            kh = st.number_input("炉膛深度系数 Kh", value=defaults['Kh'], step=0.1, on_change=trigger_furnace_recalc)

    # --- 1.3 理论计算核心 ---
    p_kva = cap_mva * 1000
    i1_theo = p_kva * 1000 / (1.732 * u1_kv * 1000)
    u2_theo = ke * (p_kva ** (1/3))
    i2_theo = p_kva * 1000 / (1.732 * u2_theo)
    
    de_theo = sqrt(i2_theo / j_den / 0.7854) * 10 # mm
    dc_theo = ky * de_theo
    di_theo = ki * de_theo
    hh_theo = kh * de_theo
    
    shell_id_theo = di_theo + 2 * lining_thick
    shell_h_theo = hh_theo + 2000

    # --- 1.4 智能圆整逻辑 ---
    if st.session_state.furnace_recalc:
        # 初次或重置时，自动填充推荐圆整值
        st.session_state.rnd_u2 = round(u2_theo)
        st.session_state.rnd_de = round(de_theo / 50) * 50 # 取整到50
        st.session_state.rnd_dc = round((st.session_state.rnd_de * ky) / 50) * 50
        st.session_state.rnd_di = round((st.session_state.rnd_de * ki) / 100) * 100
        st.session_state.rnd_hh = round((st.session_state.rnd_de * kh) / 100) * 100
        st.session_state.rnd_shell_id = st.session_state.rnd_di + 2 * lining_thick
        st.session_state.rnd_shell_h = st.session_state.rnd_hh + 2000
        st.session_state.furnace_recalc = False

    # 联动更新函数
    def update_dims():
        d = st.session_state.in_de
        st.session_state.rnd_de = d
        st.session_state.rnd_dc = round((d * ky) / 50) * 50
        st.session_state.rnd_di = round((d * ki) / 100) * 100
        st.session_state.rnd_hh = round((d * kh) / 100) * 100
        st.session_state.rnd_shell_id = st.session_state.rnd_di + 2 * lining_thick
        st.session_state.rnd_shell_h = st.session_state.rnd_hh + 2000

    with col_r:
        st.subheader("2. 设计结果与工程修正")
        
        # 结果对比表
        st.markdown("##### 📐 参数对比 (可修改右侧圆整值)")
        c1, c2, c3 = st.columns([2, 2, 2])
        c1.markdown("**参数项**")
        c2.markdown("**理论计算值**")
        c3.markdown("**工程圆整值**")
        
        # 电压电流
        c1.write("二次电压 U₂ (V)")
        c2.write(f"{u2_theo:.1f}")
        rnd_u2 = c3.number_input("设定 U₂", value=st.session_state.rnd_u2, key='in_u2', label_visibility="collapsed")
        
        rnd_i2 = p_kva * 1000 / (1.732 * rnd_u2)
        c1.write("二次电流 I₂ (A)")
        c2.write(f"{i2_theo:.0f}")
        c3.info(f"{rnd_i2:.0f}") # 反算结果
        
        # 结构参数
        c1.write("电极直径 De (mm)")
        c2.write(f"{de_theo:.0f}")
        rnd_de = c3.number_input("设定 De", value=float(st.session_state.rnd_de), step=10.0, key='in_de', on_change=update_dims, label_visibility="collapsed")
        
        c1.write("极心圆直径 Dc (mm)")
        c2.write(f"{dc_theo:.0f}")
        rnd_dc = c3.number_input("设定 Dc", value=float(st.session_state.rnd_dc), step=50.0, key='in_dc', label_visibility="collapsed")
        
        c1.write("炉膛内径 Di (mm)")
        c2.write(f"{di_theo:.0f}")
        rnd_di = c3.number_input("设定 Di", value=float(st.session_state.rnd_di), step=100.0, key='in_di', label_visibility="collapsed")
        
        c1.write("炉壳内径 (估) (mm)")
        c2.write(f"{shell_id_theo:.0f}")
        rnd_shell_id = c3.number_input("设定炉壳ID", value=float(st.session_state.rnd_shell_id), step=100.0, key='in_shell_id', label_visibility="collapsed")

        # 绘图区域
        st.markdown("---")
        st.markdown("##### 🏗️ 结构示意图 (基于圆整值)")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        # 炉壳
        rect_shell = patches.Rectangle((-rnd_shell_id/2, 0), rnd_shell_id, st.session_state.rnd_shell_h, lw=3, ec='#333', fc='none', label='炉壳')
        ax.add_patch(rect_shell)
        # 炉膛
        rect_hearth = patches.Rectangle((-rnd_di/2, 1500), rnd_di, st.session_state.rnd_hh, lw=2, ec='red', fc='#FFD700', alpha=0.3, label='熔池')
        ax.add_patch(rect_hearth)
        # 电极
        ew = rnd_de
        eh = st.session_state.rnd_shell_h * 0.7
        ax.add_patch(patches.Rectangle((-rnd_dc/2 - ew/2, st.session_state.rnd_shell_h/2), ew, eh, color='#555', label='电极'))
        ax.add_patch(patches.Rectangle((rnd_dc/2 - ew/2, st.session_state.rnd_shell_h/2), ew, eh, color='#555'))
        
        # 标注
        ax.annotate(f"炉膛内径 {rnd_di:.0f}", xy=(0, 1500 + st.session_state.rnd_hh/2), ha='center', fontsize=10, bbox=dict(fc='white', ec='none', alpha=0.7))
        ax.annotate(f"极心圆 {rnd_dc:.0f}", xy=(0, st.session_state.rnd_shell_h), xytext=(0, st.session_state.rnd_shell_h+1000), arrowprops=dict(arrowstyle='-'), ha='center')
        
        ax.set_aspect('equal')
        ax.axis('off')
        ax.legend(loc='upper right', fontsize='small')
        st.pyplot(fig)

    # --- 1.5 数据导出 ---
    st.markdown("### 📥 生成报表")
    data_export = [
        {"项目": "变压器容量", "数值": cap_mva, "单位": "MVA"},
        {"项目": "一次电压 U1", "数值": u1_kv, "单位": "kV"},
        {"项目": "一次电流 I1", "数值": round(i1_theo, 1), "单位": "A"},
        {"项目": "设计二次电压 U2", "数值": int(rnd_u2), "单位": "V"},
        {"项目": "设计二次电流 I2", "数值": int(rnd_i2), "单位": "A"},
        {"项目": "电极直径", "数值": int(rnd_de), "单位": "mm"},
        {"项目": "极心圆直径", "数值": int(rnd_dc), "单位": "mm"},
        {"项目": "炉膛内径", "数值": int(rnd_di), "单位": "mm"},
        {"项目": "炉膛深度", "数值": int(st.session_state.rnd_hh), "单位": "mm"},
        {"项目": "炉壳内径", "数值": int(rnd_shell_id), "单位": "mm"},
        {"项目": "炉壳高度", "数值": int(st.session_state.rnd_shell_h), "单位": "mm"},
        {"项目": "铜瓦配置", "数值": f"{tile_num} 块/相", "单位": "-"},
        {"项目": "铜管配置", "数值": f"{tube_num} 根/相", "单位": f"Φ{tube_d}×{tube_t}"},
    ]
    df_exp = pd.DataFrame(data_export)
    csv = df_exp.to_csv(index=False).encode('utf-8-sig')
    st.download_button("下载完整设计书 (CSV)", csv, f"矿热炉_{cap_mva}MVA_Design.csv")

# ==========================================
# 模块 2: 轴系设计 (Vol.2)
# ==========================================
elif selected_module == "🔩 轴系设计 (Vol.2)":
    st.header("🔩 传动轴设计向导")
    st.markdown("基于《机械设计手册 第2卷》，包含**强度估算**、**材料选择**与**结构设计**。")
    
    tabs = st.tabs(["1. 轴径估算", "2. 键槽选择", "3. 强度校核 (简化)"])
    
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            P_kw = st.number_input("传递功率 P (kW)", 0.1, 5000.0, 15.0)
            n_rpm = st.number_input("转速 n (r/min)", 1.0, 10000.0, 960.0)
            mat_name = st.selectbox("轴材料", list(MATERIAL_DB.keys()))
            
            # 计算逻辑
            A0 = MATERIAL_DB[mat_name]['A0']
            if n_rpm > 0:
                d_min = A0 * (P_kw / n_rpm) ** (1/3)
            else:
                d_min = 0
            
            # 考虑键槽扩大
            d_key = d_min * 1.05
            d_final = ceil(d_key / 5) * 5 # 圆整到5的倍数
            
        with c2:
            st.markdown(f"#### ✅ 计算结果")
            st.metric("扭矩 T", f"{9550*P_kw/n_rpm:.1f} N.m")
            st.metric("最小轴径 (纯扭转)", f"{d_min:.1f} mm")
            st.success(f"建议设计轴径: **Φ {d_final} mm** (已考虑键槽削弱)")
            st.caption(f"注：采用系数 A0={A0} (基于{mat_name})")

    with tabs[1]:
        st.info("根据 GB/T 1096 普通平键标准推荐")
        d_input = st.number_input("输入轴段直径 (mm)", value=int(d_final))
        b, h = recommend_key(d_input)
        t1 = h/2 + 0.2 if h > 6 else h/2 + 0.1 # 简化t1
        
        ck1, ck2, ck3 = st.columns(3)
        ck1.metric("键宽 b", f"{b} mm")
        ck2.metric("键高 h", f"{h} mm")
        ck3.metric("轴槽深 t1", f"{t1:.1f} mm")
        
        # 画截面图
        fig_shaft, ax_s = plt.subplots(figsize=(4,4))
        ax_s.add_patch(patches.Circle((0,0), d_input/2, color='#ddd', ec='black'))
        ax_s.add_patch(patches.Rectangle((-b/2, d_input/2 - t1), b, t1, color='white', ec='black'))
        ax_s.set_xlim(-d_input/1.5, d_input/1.5)
        ax_s.set_ylim(-d_input/1.5, d_input/1.5)
        ax_s.axis('off')
        ax_s.set_title("轴槽截面示意")
        st.pyplot(fig_shaft)

    with tabs[2]:
        st.warning("⚠️ 完整疲劳强度校核需要详细的受力分析图，此处仅为许用应力参考。")
        mat_info = MATERIAL_DB[mat_name]
        st.json(mat_info)

# ==========================================
# 模块 3: 齿轮传动 (Vol.3)
# ==========================================
elif selected_module == "⚙️ 齿轮传动 (Vol.3)":
    st.header("⚙️ 齿轮参数设计")
    st.markdown("基于接触强度反算模数与中心距。")
    
    col1, col2 = st.columns(2)
    with col1:
        T1 = st.number_input("小齿轮扭矩 (N.m)", value=500.0)
        u = st.number_input("传动比 u", value=3.5, step=0.1)
        beta = st.slider("螺旋角 β", 0, 30, 0, help="0为直齿")
        hardness_type = st.radio("齿面硬度", ["软齿面 (HBS<350)", "硬齿面 (HRC>55)"])
        
        # 许用应力估算
        sigma_H_lim = 600 if "软" in hardness_type else 1100
        
    with col2:
        # 试算逻辑
        Kd = 1.2 # 动载系数
        Ze = 189.8 # 钢对钢弹性系数
        Zh = 2.5 # 节点区域系数
        Phi_d = 1.0 # 齿宽系数 b/d1
        
        # 公式倒推: d1 >= ( (2KT(u+1)/u) * (Ze*Zh/sigma_H)^2 * (1/Phi_d) ) ^ (1/3)
        factor = (Ze * Zh / sigma_H_lim) ** 2
        d1_min = ( (2 * Kd * T1 * 1000 * (u+1) / u) * factor * (1/Phi_d) ) ** (1/3)
        
        # 模数估算
        z1 = 20 # 初选齿数
        m_calc = d1_min / z1
        m_std = [1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
        m_final = min([m for m in m_std if m >= m_calc], default=10)
        
        st.subheader("计算结果")
        st.metric("估算最小分度圆 d1", f"{d1_min:.2f} mm")
        st.metric("推荐模数 m", f"{m_final} mm")
        
        # 几何尺寸
        a = m_final * z1 * (1+u) / (2 * cos(radians(beta)))
        st.success(f"建议中心距 a ≈ {a:.1f} mm")
        
        # 显示详细参数
        st.table(pd.DataFrame({
            "参数": ["小齿轮齿数 z1", "大齿轮齿数 z2", "模数 m", "齿宽 b"],
            "数值": [z1, int(z1*u), m_final, int(d1_min*Phi_d)]
        }))

# ==========================================
# 模块 4: 连接紧固 (Vol.2)
# ==========================================
elif selected_module == "🔗 连接紧固 (Vol.2)":
    st.header("🔗 螺纹连接强度校核")
    
    c1, c2 = st.columns(2)
    with c1:
        load = st.number_input("轴向拉力 F (N)", value=5000.0, step=100.0)
        bolt_spec = st.selectbox("螺纹规格", THREAD_DB['d'].tolist(), index=2)
        grade = st.selectbox("性能等级", ["4.8级", "8.8级", "10.9级", "12.9级"])
        tighten = st.checkbox("需控制预紧力", value=True)
    
    with c2:
        # 获取螺纹参数
        row = THREAD_DB[THREAD_DB['d'] == bolt_spec].iloc[0]
        As = row['As']
        
        # 获取材料强度
        grade_val = float(grade.split("级")[0])
        sigma_b = int(grade_val) * 100
        sigma_s = sigma_b * (round(grade_val - int(grade_val), 1))
        
        # 计算
        # 仅受预紧力 F0, 拉力 F
        # 剩余预紧力 F'' = 1.3 F (假设)
        F_total = load * 1.3 if tighten else load
        sigma_cal = F_total / As
        
        safety = sigma_s / sigma_cal
        
        st.markdown(f"**{grade} 螺栓 M{bolt_spec}**")
        st.write(f"应力截面积 As: {As} mm²")
        st.write(f"屈服强度 σs: {sigma_s} MPa")
        
        st.divider()
        st.metric("计算应力", f"{sigma_cal:.1f} MPa")
        st.metric("安全系数 S", f"{safety:.2f}")
        
        if safety < 1.5:
            st.error("不合格！强度不足")
        elif safety > 5:
            st.warning("过度设计，建议减小规格")
        else:
            st.success("设计合格 ✅")

# ==========================================
# 模块 5: 综合查询
# ==========================================
elif selected_module == "📚 综合查询":
    st.header("📚 设计数据速查")
    st.markdown("直接调用后台数据库，无需翻书。")
    
    q_type = st.selectbox("查询类别", ["常用材料性能", "普通螺纹尺寸", "矿热炉经验系数"])
    
    if q_type == "常用材料性能":
        df = pd.DataFrame(MATERIAL_DB).T
        st.dataframe(df, use_container_width=True)
    elif q_type == "普通螺纹尺寸":
        st.dataframe(THREAD_DB, use_container_width=True)
    elif q_type == "矿热炉经验系数":
        df = pd.DataFrame(FURNACE_DB).T
        st.dataframe(df, use_container_width=True)