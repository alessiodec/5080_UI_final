import streamlit as st

# Set page configuration first
st.set_page_config(layout="wide")

# SIDEBAR NAVIGATION
st.sidebar.title("Navigation")
# --- Removed "Test" and "Symbolic Regression" from the list ---
page = st.sidebar.radio("Go to", [
    "Home",
    "Data Analysis // Surrogate Modelling",
    "Optimisation",
    "Symbolic Regression (BETA)"
])

# ROUTE TO PAGE
if page == "Data Analysis // Surrogate Modelling":
    import data_analysis
    data_analysis.data_analysis()
elif page == "Optimisation":
    import optimisation
    optimisation.run()
# --- Removed the elif block for "Symbolic Regression" ---
elif page == "Symbolic Regression (BETA)":
    # This now correctly points to your symbolic_regression.py module
    import symbolic_regression
    symbolic_regression.run()
# --- Removed the elif block for "Test" ---
else: # This defaults to the Home page
    # HOME PAGE TEXT
    st.title("Machine Learning-Enabled Optimisation of Industrial Flow Systems - UI Tool")
    st.write("""
This tool will guide you through the ML framework developed to analyse two industrial energy flow systems.
The framework is composed of three sections: surrogate modelling, multi-objective optimisation, and symbolic regression.
 - **Surrogate Modelling**: Constructing a computationally efficient metamodel using neural networks to approximate complex system behaviour.
 - **Multi-Objective Optimisation**: Balancing conflicting objectives to achieve a trade-off solution.
 - **Symbollic Regression**: Utilises methods like genetic programming to automatically generate interpretable mathematical expressions that accurately capture the underlying data relationships.
The first flow system is for a heatsink, where the available dataset consists of two geometric input parameters that yield outputs for pressure drop and thermal resistance.
The second flow system involves the multi-physics software Leeds COMSOL, used to predict corrosion rate and saturation ratio in geothermal steel pipes for five inputs.
The aim of this UI is to incorporate the customisable areas of this project to allow the user to gain an enhanced understanding of the methods at hand.
**Please select a page from the sidebar.**
*note: in the event of no response from pressing any buttons, press twice to move to the desired page, or refresh the app to go back to home.*
    """)
