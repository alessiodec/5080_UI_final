import streamlit as st

# Define pages and their descriptions.
pages = {
    "Home": "This is the main page where you get an overview of the project.",
    "Data Analysis": "This page performs data analysis on your data.",
    "Optimisation": "This page runs optimisation routines on your model.",
    "Physical Relationship Analysis": "This page examines physical relationship data."
}

# Inject custom CSS to style the horizontal radio as a tab bar.
tab_css = """
<style>
/* Make the radio buttons horizontal */
div.stRadio > div { 
    display: flex; 
    flex-direction: row;
    justify-content: flex-start;
}
/* Style the labels to look like tabs */
div.stRadio > div > label {
    margin-right: 20px;
    padding: 10px 20px;
    border: 1px solid #ddd;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    background-color: #f5f5f5;
    cursor: pointer;
}
/* Hover effect */
div.stRadio > div > label:hover {
    background-color: #e5e5e5;
}
/* Hide the default radio button */
div.stRadio > div > input[type="radio"] {
    display: none;
}
/* Highlight the selected tab by using the adjacent sibling selector workaround.
   Streamlit doesn't allow direct styling of the checked state, so we simulate this by re-rendering the radio
   with the selected value having a different background.
*/
</style>
"""
st.markdown(tab_css, unsafe_allow_html=True)

# Create horizontal "tab" navigation using st.radio.
# Using horizontal=True forces a row layout.
page = st.radio("", list(pages.keys()), horizontal=True)

# Render the main content based on the selected page.
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
    st.write("Please select a page from the tab navigation above.")

# Add a sidebar area to display the description of the current page.
st.sidebar.title("Page Description")
st.sidebar.write(pages.get(page, ""))
