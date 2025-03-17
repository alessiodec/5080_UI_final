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
    st.write("""
    Welcome! This tool guides you through the ML framework developed to analyse two industrial energy flow systems.
    The framework comprises three key sections: surrogate modelling, multi-objective optimisation, and symbolic regression.
    The first flow system pertains to a heatsink, with a dataset containing two geometric input parameters that yield outputs for pressure drop and thermal resistance.
    The second flow system utilises the multi-physics software Leeds COMSOL to predict corrosion rate and saturation ratio in geothermal steel pipes.
    Please select a page from the sidebar.
    """)
