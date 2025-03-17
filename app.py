import streamlit as st

# Set page configuration first
st.set_page_config(layout="wide")

# SIDEBAR NAVIGATION
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Optimisation", "Physical Relationship Analysis"])

# ROUTE TO PAGE
if page == "Data Analysis":
    import data_analysis
    data_analysis.data_analysis()
elif page == "Optimisation":
    import optimisation
    optimisation.run()
elif page == "Physical Relationship Analysis":
    import physical_relationship_analysis
    physical_relationship_analysis.run()
else:
    # HOME PAGE TEXT
    st.title("Machine Learning-Enabled Optimisation of Industrial Flow Systems - UI Tool")
    st.write("This tool will guide you through the ML framework developed to analyse two industrial energy flow systems.")
    st.write("The framework is composed of three sections: surrogate modelling, multi-objective optimisation, and symbollic regression.")
    st.write("The first flow system is a for a heatsink, where the avaliable dataset consists of two geoemetric input parameters, and outputting pressure drop and thermal resistance.")
    st.write("The second flow system is that of a multi-physics software, Leeds COMSOL, used to predict corrosion rate and saturation ratio in gheothermal steel pipes. ")
    st.write("Please select a page from the sidebar.")
