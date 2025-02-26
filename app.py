import streamlit as st

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Optimisation", "Physical Relationship Analysis"])

if page == "Data Analysis":
    st.session_state["data_analysis_page"] = "main"  # Reset Data Analysis state
    import data_analysis
    data_analysis.data_analysis()
elif page == "Optimisation":
    st.session_state["optimisation_page"] = "optimisation"  # Reset Optimisation state
    import optimisation
    optimisation.run()
elif page == "Physical Relationship Analysis":
    st.session_state["physical_relationship_page"] = "Main Menu"  # Reset Physical Relationship Analysis state
    import physical_relationship_analysis
    physical_relationship_analysis.run()
else:
    st.title("Welcome to the Main Page")
    st.write("Please select a page from the sidebar.")
