import streamlit as st
import os

st.markdown("# About the Team")
st.write("Meet the individuals behind GDPredict!")

# Define team member details
team_members = [
    {
        "name": "Katie Huang",
        "role": "Coder",
        "description": """Data + Back end""", 
        "image": "Katie.png"
    },
    {
        "name": "Diya Jibu",
        "role": "Coder",
        "description": """Data + Back end""",
        "image": "Diya.png"
    },
    {
        "name": "Kayla Nguyen",
        "role": "Scribe", 
        "description": """Design + Front end""",
        "image": "Kayla.png"
    },
    {
        "name": "Evelyn Tran",
        "role": "Group Leader",
        "description": """Design + Front end""",
        "image": "Evelyn.png"
    }
]

# Display each team member in a row
for member in team_members:
    # Create a container for consistent spacing
    with st.container():
        # Create columns for photo and text within the full-width container
        img_col, text_col = st.columns([1, 3])
        
        # Image on the left
        with img_col:
            if os.path.exists(f"assets/{member['image']}"):
                st.image(f"assets/{member['image']}", width=150)
            else:
                st.warning(f"Please add {member['image']} to the assets folder")
        
        # Text content on the right
        with text_col:
            # Create a div with custom styling for all text content
            st.markdown(f'''
                <div style="line-height: 1.2;">
                    <h2 style="font-size: 28px; font-weight: bold; margin-bottom: 2px;">{member["name"]}</h2>
                    <p style="font-size: 16px; font-style: italic; margin: 0 0 12px 0;">{member["role"]}</p>
                    <p style="font-size: 16px; margin: 0;">{member["description"]}</p>
                </div>
            ''', unsafe_allow_html=True)
        
        # Add divider between team members
            st.write("---")# Optional: Add a footer about the team
st.write(
    """We are a team from the Univeristy of Texas at Dallas studying Computer Science!"""
)
