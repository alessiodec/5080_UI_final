import streamlit as st

# Initialize current page if not set
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Sidebar Navigation with a key for tracking changes
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
                        ["Home", "Data Analysis", "Optimisation", "Physical Relationship Analysis"],
                        key="nav")

# When page changes, clear only page-specific keys
if page != st.session_state.current_page:
    # Define keys to keep (like navigation state)
    keys_to_keep = {"nav", "current_page"}
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            st.session_state.pop(key)
    st.session_state.current_page = page
    st.experimental_rerun()

# Load pages based on current selection
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
