import streamlit as st

st.markdown("<style>.stApp { background-color: white; }</style>", unsafe_allow_html=True)
st.set_page_config(layout="wide")

# SIDEBAR NAVIGATION
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Optimisation", "Physical Relationship Analysis"])

# ROUTE TO PAGE
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
    # HOME PAGE TEXT
    st.title("Machine Learning-Enabled Optimisation of Industrial Flow Systems - UI Tool")
    st.write("This UI incorporates the customisable areas of this project for a user to design and solve their own problems.")
    st.write("Please select a page from the sidebar.")
