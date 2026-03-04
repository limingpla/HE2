import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

# 设置页面配置
st.set_page_config(page_title="Cognitive Impairment Prediction", layout="wide")

# 添加错误处理的模型加载
@st.cache_resource
def load_model():
    """加载模型并添加错误处理"""
    try:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'RF.pkl')
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            # 尝试其他路径
            alternative_paths = [
                'RF.pkl',
                './RF.pkl',
                os.path.join(os.getcwd(), 'RF.pkl')
            ]
            
            for path in alternative_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            else:
                st.error(f"找不到模型文件。当前目录: {os.getcwd()}")
                st.write("目录中的文件:", os.listdir('.'))
                st.stop()
        
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        st.stop()

# 加载模型
model = load_model()

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "D_D": {"type": "numerical", "min": 0.18, "max": 47.17, "default": 2.0},
    "CRP": {"type": "numerical", "min": 3.2, "max": 360, "default": 10.0},
    "INR": {"type": "numerical", "min": 0.91, "max": 8.99, "default": 2.0},
    "WBC": {"type": "numerical", "min": 0.1, "max": 133.67, "default": 10.0},
    "FDP": {"type": "numerical", "min": 0.23, "max": 195.8, "default": 12.0},
    "Lactate": {"type": "numerical", "min": 0.3, "max": 20, "default": 4.0},
    "PTA": {"type": "numerical", "min": 9, "max": 143, "default": 40.0},
    "APTT": {"type": "numerical", "min": 20, "max": 180, "default": 80.0},
}

# Streamlit 界面
st.title("🧠 Cognitive Impairment Prediction Model")
st.markdown("---")

# 侧边栏输入
with st.sidebar:
    st.header("📊 Input Features")
    st.markdown("Enter the following feature values:")
    
    feature_values = []
    for feature, properties in feature_ranges.items():
        if properties["type"] == "numerical":
            value = st.number_input(
                label=f"**{feature}**",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                format="%.2f",
                help=f"Range: {properties['min']} - {properties['max']}"
            )
            feature_values.append(value)

# 主界面
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Summary")
    input_df = pd.DataFrame([feature_values], columns=list(feature_ranges.keys()))
    st.dataframe(input_df, use_container_width=True)

# 预测按钮
if st.sidebar.button("🔍 Predict", type="primary", use_container_width=True):
    with st.spinner("Calculating prediction and SHAP values..."):
        try:
            # 转换为模型输入格式
            feature_df = pd.DataFrame([feature_values], columns=list(feature_ranges.keys()))
            
            # 模型预测
            predicted_class = model.predict(feature_df)[0]
            predicted_proba = model.predict_proba(feature_df)[0]
            
            # 获取类别标签（假设二分类）
            class_labels = ["No Impairment", "Cognitive Impairment"]
            predicted_label = class_labels[predicted_class]
            
            # 提取预测的类别概率
            probability = predicted_proba[predicted_class] * 100
            
            with col2:
                st.subheader("🎯 Prediction Result")
                
                # 显示预测结果
                if predicted_class == 1:
                    st.error(f"### ⚠️ {predicted_label}")
                else:
                    st.success(f"### ✅ {predicted_label}")
                
                # 显示概率
                st.metric("Probability", f"{probability:.1f}%")
                
                # 显示所有类别的概率
                st.write("**Class Probabilities:**")
                prob_df = pd.DataFrame({
                    'Class': class_labels,
                    'Probability': [f"{p*100:.1f}%" for p in predicted_proba]
                })
                st.dataframe(prob_df, use_container_width=True)
            
            # SHAP 分析部分
            st.markdown("---")
            st.subheader("🔬 SHAP Analysis")
            
            try:
                # 计算 SHAP 值
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(feature_df)
                
                # 处理 SHAP 值（针对不同的输出格式）
                if isinstance(shap_values, list):
                    # 多分类情况
                    shap_plot_values = shap_values[predicted_class]
                    expected_value = explainer.expected_value[predicted_class]
                else:
                    # 二分类或回归
                    shap_plot_values = shap_values
                    expected_value = explainer.expected_value
                
                # 创建两列布局
                col_shap1, col_shap2 = st.columns([1, 1])
                
                with col_shap1:
                    st.write("**Force Plot**")
                    # 生成 SHAP 力图
                    fig, ax = plt.subplots(figsize=(10, 3))
                    shap.force_plot(
                        expected_value,
                        shap_plot_values[0],  # 第一个样本
                        feature_df.iloc[0],
                        matplotlib=True,
                        show=False,
                        figsize=(10, 3)
                    )
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with col_shap2:
                    st.write("**Feature Importance**")
                    # 创建特征重要性条形图
                    if isinstance(shap_plot_values, np.ndarray):
                        shap_importance = np.abs(shap_plot_values[0])
                        feature_imp = pd.DataFrame({
                            'Feature': list(feature_ranges.keys()),
                            '|SHAP Value|': shap_importance
                        }).sort_values('|SHAP Value|', ascending=True)
                        
                        fig2, ax2 = plt.subplots(figsize=(8, 4))
                        ax2.barh(feature_imp['Feature'], feature_imp['|SHAP Value|'])
                        ax2.set_xlabel('|SHAP Value| (Impact on Prediction)')
                        ax2.set_title('Feature Impact')
                        plt.tight_layout()
                        st.pyplot(fig2)
                        plt.close()
                
                # 显示详细的SHAP值
                with st.expander("📋 View Detailed SHAP Values"):
                    shap_df = pd.DataFrame({
                        'Feature': list(feature_ranges.keys()),
                        'Feature Value': feature_values,
                        'SHAP Value': shap_plot_values[0] if isinstance(shap_plot_values, np.ndarray) else shap_plot_values
                    })
                    st.dataframe(shap_df, use_container_width=True)
                    
            except Exception as e:
                st.warning(f"SHAP visualization unavailable: {str(e)}")
                
                # 备用方案：显示特征重要性
                if hasattr(model, 'feature_importances_'):
                    st.subheader("📊 Feature Importance (Alternative)")
                    feature_importance = pd.DataFrame({
                        'Feature': list(feature_ranges.keys()),
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=True)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'])
                    ax.set_xlabel('Feature Importance')
                    ax.set_title('Random Forest Feature Importance')
                    
                    # 添加数值标签
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width, bar.get_y() + bar.get_height()/2, 
                               f'{width:.3f}', ha='left', va='center', fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.exception(e)

# 添加页脚
st.markdown("---")
st.markdown("📌 *This is a machine learning prediction tool. Please consult healthcare professionals for clinical decisions.*")
