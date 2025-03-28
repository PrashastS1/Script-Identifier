import pandas as pd
import plotly.express as px
import os

def load_data(csv_path):
    """Loads dataset CSV and extracts script counts."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path, header=None, names=["image_path", "word", "script"])
    return df["script"].value_counts()

def plot_script_distribution(script_counts, title="Script Distribution in Dataset"):
    """Plots the distribution of scripts using Plotly."""
    fig = px.bar(
        x=script_counts.index,
        y=script_counts.values,
        labels={"x": "Script Language", "y": "Count"},
        title=title,
        text_auto=True
    )
    fig.show()

def main():
    """Main function to process both train and test datasets."""
    train_csv_path =  r"C:\Users\jenis_td7jjpo\Desktop\PRML\Project\Script-Identifier\data\recognition\train.csv" # Update with actual path
    test_csv_path = r"C:\Users\jenis_td7jjpo\Desktop\PRML\Project\Script-Identifier\data\recognition\test.csv"    # Update with actual path
    
    try:
        train_script_counts = load_data(train_csv_path)
        test_script_counts = load_data(test_csv_path)
        
        print("Train Set Script Counts:")
        print(train_script_counts)  # Print distribution
        plot_script_distribution(train_script_counts, "Train Set Script Distribution")
        
        print("Test Set Script Counts:")
        print(test_script_counts)  # Print distribution
        plot_script_distribution(test_script_counts, "Test Set Script Distribution")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
