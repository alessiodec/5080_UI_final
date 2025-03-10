import streamlit as st

# Dictionary holding page descriptions
page_info = {
    "Home": "This is the main landing page. You can design and solve your own problems here.",
    "Data Analysis": "This page provides data analysis tools and visualizations.",
    "Optimisation": "This page allows you to run optimisation algorithms.",
    "Physical Relationship Analysis": "This page contains tools to analyse physical relationships."
}

# Use the built-in sidebar for both navigation and page info
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Home", "Data Analysis", "Optimisation", "Physical Relationship Analysis"])
    st.markdown("### Page Information")
    st.write(page_info[page])

# Main content area
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
    st.write("This UI incorporates the customizable areas of this project for a user to design and solve their own problems.")
    st.write("Please select a page from the sidebar.")
