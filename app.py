import streamlit as st

# Define the pages and their descriptions.
pages = {
    "Home": "This is the main page where you get an overview of the project.",
    "Data Analysis": "This page performs data analysis on your data.",
    "Optimisation": "This page runs optimisation routines on your model.",
    "Physical Relationship Analysis": "This page examines physical relationship data."
}

# Initialize the session state for current_page if not already set.
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Create a row of buttons for navigation (simulating top tabs).
cols = st.columns(len(pages))
for i, page_name in enumerate(pages.keys()):
    if cols[i].button(page_name):
        st.session_state.current_page = page_name

# Determine the current page from the session state.
page = st.session_state.current_page

# Display the main content based on the selected page.
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
    # Home page content.
    st.title("Welcome to the Main Page")
    st.write("This UI incorporates the customizable areas of this project for a user to design and solve their own problems.")
    st.write("Please select a page from the top navigation.")

# Add a sidebar area to display a description of the current page.
st.sidebar.title("Page Description")
st.sidebar.write(pages.get(page, ""))
