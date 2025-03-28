import pandas as pd
import plotly.express as px

# Load Dataset
train_csv_path = r"C:\Users\jenis_td7jjpo\Desktop\PRML\Project\Script-Identifier\data\recognition\train.csv"  # Update path if needed
df = pd.read_csv(train_csv_path)

# Count occurrences of each script
script_counts = df['Language'].value_counts()

# Bar Chart
fig_bar = px.bar(
    script_counts,
    x=script_counts.index,
    y=script_counts.values,
    title="Script Distribution in Training Set",
    labels={'x': 'Script', 'y': 'Count'},
    text_auto=True
)
fig_bar.show()

# Pie Chart
fig_pie = px.pie(
    names=script_counts.index,
    values=script_counts.values,
    title="Script Distribution (Proportion)",
    hole=0.3
)
fig_pie.show()
