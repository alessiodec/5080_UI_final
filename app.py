import streamlit as st

# Initialize current page in session state if not already set
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Sidebar Navigation with a key to track changes
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
                        ["Home", "Data Analysis", "Optimisation", "Physical Relationship Analysis"],
                        key="nav")

# Reset session state if the page has changed
if page != st.session_state.current_page:
    # Optionally, clear only specific keys instead of the entire state.
    st.session_state.clear()
    st.session_state.current_page = page
    st.experimental_rerun()

# Load pages based on the current selection
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
    st.write("This UI incorporates the customisable areas of this project for a user to design and solve their own problems")
    st.write("Please select a page from the sidebar.")
