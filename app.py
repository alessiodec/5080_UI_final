import streamlit as st

# Dictionary holding page descriptions
page_info = {
    "Home": "This is the main landing page. You can design and solve your own problems here.",
    "Data Analysis": "This page provides data analysis tools and visualizations.",
    "Optimisation": "This page allows you to run optimisation algorithms.",
    "Physical Relationship Analysis": "This page contains tools to analyse physical relationships."
}

# Create three columns:
# - col_nav for the navigation sidebar
# - col_info for the page information sidebar
# - col_main for the main content area
col_nav, col_info, col_main = st.columns([1, 1, 3])

# Navigation Sidebar (Leftmost Column)
with col_nav:
    st.title("Navigation")
    page = st.radio("Go to", ["Home", "Data Analysis", "Optimisation", "Physical Relationship Analysis"])

# Page Information Sidebar (Middle Column)
with col_info:
    st.markdown("### Page Information")
    st.write(page_info[page])

# Main Content Area (Rightmost Column)
with col_main:
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
        st.write("Please select a page from the navigation.")
