import streamlit as st

# Set page configuration first
st.set_page_config(layout="wide")

# SIDEBAR NAVIGATION
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis // Surrogate Modelling", "Optimisation", "Symbollic Regression"])

# ROUTE TO PAGE
if page == "Data Analysis // Surrogate Modelling":
    import data_analysis
    data_analysis.data_analysis()
elif page == "Optimisation":
    import optimisation
    optimisation.run()
elif page == "Symbollic Regression":
    import physical_relationship_analysis
    physical_relationship_analysis.run()
else:
    # HOME PAGE TEXT
    st.title("Machine Learning-Enabled Optimisation of Industrial Flow Systems - UI Tool")
    st.write("""
This tool will guide you through the ML framework developed to analyse two industrial energy flow systems.\n
The framework is composed of three sections: surrogate modelling, multi-objective optimisation, and symbolic regression.\n
 - **Surrogate Modelling**: Constructing a computationally efficient metamodel using neural networks to approximate complex system behaviour.\n
 - **Multi-Objective Optimisation**: Balancing conflicting objectives to achieve a trade-off solution.\n
 - **Symbollic Regression**: Utilises methods like genetic programming to automatically generate interpretable mathematical expressions that accurately capture the underlying data relationships. \n
The first flow system is for a heatsink, where the available dataset consists of two geometric input parameters that yield outputs for pressure drop and thermal resistance.\n
The second flow system involves the multi-physics software Leeds COMSOL, used to predict corrosion rate and saturation ratio in geothermal steel pipes for five inputs.\n
The aim of this UI is to incorporate the customisable areas of this project to allow the user to gain an ehnaced understanding of the methods at hand.
Please select a page from the sidebar.
    """)
