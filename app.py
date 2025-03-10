import streamlit as st

# Define pages and their descriptions.
pages = {
    "Home": "This is the main page where you get an overview of the project.",
    "Data Analysis": "This page performs data analysis on your data.",
    "Optimisation": "This page runs optimisation routines on your model.",
    "Physical Relationship Analysis": "This page examines physical relationship data."
}

# Get the current page from query parameters (default to Home).
query_params = st.query_params
if "page" in query_params and query_params["page"]:
    current_page = query_params["page"][0]
else:
    current_page = "Home"

# Generate the tab ribbon HTML.
# Each link uses an onclick event to update the URL in the same window.
tabs_html = '<div class="tab">'
for page in pages:
    active_class = "active" if page == current_page else ""
    # Use href="javascript:void(0)" so clicking doesn't trigger default link behavior,
    # then onclick sets window.location.href to update the query parameter.
    tabs_html += (
        f'<a class="tablinks {active_class}" href="javascript:void(0)" '
        f'onclick="window.location.href=\'?page={page}\'">{page}</a>'
    )
tabs_html += '</div>'

# CSS styling for the tab ribbon.
tab_css = """
<style>
.tab {
  overflow: hidden;
  border-bottom: 1px solid #ccc;
  background-color: #f1f1f1;
  margin-bottom: 20px;
}
.tab a {
  float: left;
  border: 1px solid transparent;
  border-bottom: none;
  background-color: inherit;
  padding: 14px 16px;
  text-decoration: none;
  font-size: 17px;
  color: black;
  transition: 0.3s;
}
.tab a:hover {
  background-color: #ddd;
}
.tab a.active {
  border: 1px solid #ccc;
  border-bottom: 1px solid white;
  background-color: white;
}
</style>
"""

# Render the tab ribbon.
st.markdown(tab_css + tabs_html, unsafe_allow_html=True)

# Render the main content based on the current page.
if current_page == "Data Analysis":
    import data_analysis
    data_analysis.data_analysis()
elif current_page == "Optimisation":
    import optimisation
    optimisation.run()
elif current_page == "Physical Relationship Analysis":
    import physical_relationship_analysis
    physical_relationship_analysis.run()
else:
    st.title("Welcome to the Main Page")
    st.write("This UI incorporates the customizable areas of this project for a user to design and solve their own problems.")
    st.write("Please select a page from the tab ribbon above.")

# Sidebar: Display the description for the current page.
st.sidebar.title("Page Description")
st.sidebar.write(pages.get(current_page, ""))
