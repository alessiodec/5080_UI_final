import streamlit as st

# SIDEBAR NAVIGATION
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Optimisation", "Physical Relationship Analysis"])

# PAGE INFORMATION DICTIONARY
# This dictionary maps each page to a description.
page_info = {
    "Home": "This is the main landing page. You can design and solve your own problems here.",
    "Data Analysis": "This page provides data analysis tools and visualizations.",
    "Optimisation": "This page allows you to run optimisation algorithms.",
    "Physical Relationship Analysis": "This page contains tools to analyse physical relationships."
}

# ADDITIONAL SIDEBAR FOR PAGE INFORMATION
st.sidebar.markdown("### Page Information")
st.sidebar.write(page_info[page])

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
    st.title("Welcome to the Main Page")
    st.write("This UI incorporates the customizable areas of this project for a user to design and solve their own problems.")
    st.write("Please select a page from the sidebar.")
