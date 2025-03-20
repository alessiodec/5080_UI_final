import streamlit as st

def run():
    # Display the prompt text at the top of the page.
    st.write("please select a dataset:")

    # Display a button labeled "corrosion".
    if st.button("corrosion"):
        # Placeholder for when the corrosion dataset is selected.
        st.write("Corrosion dataset selected. [Further implementation goes here]")
    
    # Display a button labeled "heatsink".
    if st.button("heatsink"):
        # Placeholder for when the heatsink dataset is selected.
        st.write("Heatsink dataset selected. [Further implementation goes here]")
    
    # Display a button labeled "go to home" that navigates back to the main page.
    if st.button("go to home"):
        # Set a query parameter 'page' to "Home" and re-run the app.
        # This effectively returns the user to the main page in your multi-page app.
        st.experimental_set_query_params(page="Home")
        st.experimental_rerun()
