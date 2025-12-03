import streamlit as st
import pandas as pd
import plotly.graph_objects as go # This is the line that caused the error

# --- Application Logic ---
st.title("Ice Cream Sales Dashboard")

# Create a sample DataFrame for demonstration
data = {
    'Flavor': ['Vanilla', 'Chocolate', 'Strawberry', 'Mint Chip'],
    'Sales': [150, 220, 90, 110]
}
df = pd.DataFrame(data)

# Create a Plotly bar chart
fig = go.Figure(data=[
    go.Bar(
        x=df['Flavor'],
        y=df['Sales'],
        marker_color=['gold', 'saddlebrown', 'red', 'lightgreen']
    )
])

fig.update_layout(
    title='Ice Cream Sales by Flavor',
    xaxis_title='Flavor',
    yaxis_title='Units Sold'
)

# Display the Plotly chart in Streamlit
st.plotly_chart(fig, use_container_width=True)
