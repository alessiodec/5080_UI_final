import streamlit as st

def run():
    # Display the prompt text at the top of the page.
    st.write("Please Select a Dataset:")

    # Create two columns to place the dataset buttons beside each other.
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Corrosion"):
            st.write("Corrosion dataset selected. [Further implementation goes here]")
    
    with col2:
        if st.button("Heatsink"):
            st.write("Heatsink dataset selected. [Further implementation goes here]")
    
    # Place the "Go To Home" button beneath the columns.
    if st.button("Go To Home"):
        st.experimental_set_query_params(page="Home")
        st.experimental_rerun()
