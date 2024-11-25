import streamlit as st

from gateway import query


# TODO: change roles and add app description
#


role_options = ["admin", "manager", "employee"]

roles = ["all"]
roles += st.pills("Roles", role_options, selection_mode="multi")

# print(roles)

question = st.text_input("Enter a question")

answer = query(question, roles)

if st.button("Submit"):
    st.write(answer)
