import streamlit as st
from utils.predict import predict_disease, predict_disease_detailed
from PIL import Image
import tempfile
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
        color: #000000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box h3 {
        color: #000000 !important;
        font-weight: bold;
    }
    .prediction-box p {
        color: #000000 !important;
        font-weight: 500;
    }
    .healthy-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        color: #000000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .disease-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        color: #000000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: #000000 !important;
        font-weight: 500;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .info-box strong {
        color: #000000 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🌿 Plant Disease Classifier</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Application Info")
        st.info("""
        This application uses deep learning to identify plant diseases from leaf images.
        
        **Supported formats:** JPG, JPEG, PNG
        
        **Supported plants:** Apple, Corn, Grape, Tomato, Potato, and more!
        """)
        
        st.header("🔧 Settings")
        show_detailed_results = st.checkbox("Show detailed analysis", value=True)
        show_top_predictions = st.checkbox("Show top 3 predictions", value=True)
        
        st.header("📈 Statistics")
        if 'prediction_count' not in st.session_state:
            st.session_state.prediction_count = 0
        st.metric("Total Predictions", st.session_state.prediction_count)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a leaf image...", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a plant leaf for disease detection"
        )
        
        if uploaded_file:
            # Display uploaded image
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, caption="Uploaded Leaf Image", use_container_width=True)
            
            # Image information
            st.markdown("**Image Information:**")
            st.write(f"- **Filename:** {uploaded_file.name}")
            st.write(f"- **Size:** {image_pil.size}")
            st.write(f"- **Format:** {image_pil.format}")
            st.write(f"- **Mode:** {image_pil.mode}")
    
    with col2:
        if uploaded_file:
            st.header("🔍 Analysis Results")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image_pil.save(tmp_file.name)
                tmp_file_path = tmp_file.name
            
            try:
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Analyzing image...")
                progress_bar.progress(25)
                
                if show_detailed_results:
                    # Detailed prediction
                    class_name, confidence, prediction_details, disease_info = predict_disease_detailed(tmp_file_path)
                else:
                    # Simple prediction
                    class_name, confidence = predict_disease(tmp_file_path)
                    disease_info = None
                    prediction_details = None
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                # Update prediction count
                st.session_state.prediction_count += 1
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                # Main prediction result
                is_healthy = 'healthy' in class_name.lower()
                box_class = "healthy-box" if is_healthy else "disease-box"
                
                formatted_name = class_name.replace('_', ' ').title()
                confidence_percentage = confidence * 100
                
                st.markdown(f"""
                <div class="prediction-box {box_class}">
                    <h3>🩺 Diagnosis: {formatted_name}</h3>
                    <p><strong>Confidence:</strong> {confidence_percentage:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence meter
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence_percentage,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Confidence Level", 'font': {'size': 16, 'color': '#6c757d'}},
                    number={'font': {'size': 24, 'color': '#6c757d'}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickcolor': '#adb5bd', 'tickfont': {'color': '#6c757d'}},
                        'bar': {'color': "#ff6347" if confidence < 0.6 else "#ffa500" if confidence < 0.8 else "#32cd32"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#adb5bd",
                        'steps': [
                            {'range': [0, 60], 'color': "#ffebee"},
                            {'range': [60, 80], 'color': "#fff3e0"},
                            {'range': [80, 100], 'color': "#e8f5e8"}
                        ],
                        'threshold': {
                            'line': {'color': "#6c757d", 'width': 3},
                            'thickness': 0.75,
                            'value': 85
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#6c757d'}
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Detailed information
                if show_detailed_results and disease_info:
                    st.subheader("📋 Detailed Information")
                    
                    col1_detail, col2_detail = st.columns(2)
                    
                    with col1_detail:
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>Plant Type:</strong> {disease_info['plant_type']}<br>
                            <strong>Condition:</strong> {disease_info['condition']}<br>
                            <strong>Health Status:</strong> {'Healthy' if disease_info['is_healthy'] else 'Disease Detected'}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2_detail:
                        status_icon = "✅" if disease_info['is_healthy'] else "⚠️"
                        status_color = "green" if disease_info['is_healthy'] else "red"
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>Status:</strong> <span style="color: {status_color}">{status_icon} {disease_info['condition']}</span><br>
                            <strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Top predictions chart
                if show_top_predictions and show_detailed_results and prediction_details:
                    st.subheader("📊 Top Predictions")
                    
                    top_preds = prediction_details['top_predictions']
                    labels = [name.replace('_', ' ').title() for name in top_preds.keys()]
                    values = [conf * 100 for conf in top_preds.values()]
                    
                    fig_bar = px.bar(
                        x=labels,
                        y=values,
                        title="Top 3 Prediction Confidence Scores",
                        labels={'x': 'Disease/Condition', 'y': 'Confidence (%)'},
                        color=values,
                        color_continuous_scale='Viridis'
                    )
                    fig_bar.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 Recommendations")
                if is_healthy:
                    st.success("""
                    🎉 **Great news!** Your plant appears to be healthy. 
                    
                    **Maintenance tips:**
                    - Continue regular watering and proper lighting
                    - Monitor for any changes in leaf appearance
                    - Maintain good air circulation around the plant
                    """)
                else:
                    st.warning(f"""
                    ⚠️ **Disease detected:** {formatted_name}
                    
                    **Recommended actions:**
                    - Isolate the affected plant to prevent spread
                    - Consult with a plant pathologist or agriculture expert
                    - Consider appropriate treatment options
                    - Monitor other plants for similar symptoms
                    
                    **Note:** This is an AI prediction. Please consult with experts for proper diagnosis and treatment.
                    """)
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.info("Please try uploading a different image or contact support if the problem persists.")
            
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
        else:
            st.info("👆 Please upload an image to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🌿 Plant Disease Classifier | Built with Streamlit & TensorFlow</p>
        <p><em>For educational and research purposes. Always consult with agricultural experts for professional advice.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
