import streamlit as st

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Optimisation", "Physical Relationship Analysis"])

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
    st.title("Welcome to the Main Page")
    st.write("Please select a page from the sidebar.")
