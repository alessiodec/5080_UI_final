import streamlit as st

# LEFT SIDEBAR: Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Optimisation", "Physical Relationship Analysis"])

# Dictionary holding page descriptions
page_info = {
    "Home": "This is the main landing page. You can design and solve your own problems here.",
    "Data Analysis": "This page provides data analysis tools and visualizations.",
    "Optimisation": "This page allows you to run optimisation algorithms.",
    "Physical Relationship Analysis": "This page contains tools to analyse physical relationships."
}

# MAIN LAYOUT: Two columns
# The first column (col_main) is for page content.
# The second column (col_sidebar) simulates a second sidebar.
col_main, col_sidebar = st.columns([3, 1])

# RIGHT SIDEBAR SIMULATION: Display additional page information
with col_sidebar:
    st.markdown("### Page Information")
    st.write(page_info[page])

# MAIN CONTENT: Render page based on selection
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
        st.write("Please select a page from the sidebar.")
