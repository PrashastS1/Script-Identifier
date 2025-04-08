import gradio as gr
import os

# Function to process the image (placeholder)
def process_image(img, language):
    # This is where you would implement your OCR logic
    # For now, it's just a placeholder that returns the image and language
    return f"Selected language: {language}"

# Set theme and layout
theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#10171e",
    block_background_fill="#1a2633",
    background_fill_primary="#1e2a38",
    block_label_text_color="white",
    button_primary_background_fill="#2a5d84",
    button_primary_background_fill_hover="#3a6d94",
    button_secondary_background_fill="#475569",
    button_secondary_background_fill_hover="#576579",
)

# Create the Gradio interface
with gr.Blocks(theme=theme) as demo:
    # Header with logos and title
    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            gr.HTML("""
                <div style="text-align: center; color: white;">
                    <h1 style="margin-bottom: 0;">Script Identification</h1>
                    <h2 style="margin-top: 0.5em;">Developed by Tera bhai Seedhe Maut</h2>
                </div>
            """)
            
    with gr.Row(equal_height=True):
        with gr.Column(scale=5, min_width=100):
            gr.Image("assets/IITJ_logo.png", show_label=False, container=False, height=155, width=120)

        with gr.Column(scale=1, min_width=100):
            gr.Image("assets/masti.png", show_label=False, container=False, height=155, width=120)

    # Repository links
    with gr.Row(equal_height=True):
        with gr.Column():
            gr.HTML("""
                <div style="text-align: center; margin: 20px 0;">
                    <a href="https://github.com/AurindumBanerjee/Script-Identifier" style="color: #5e9ed6; margin: 0 15px; text-decoration: none; font-size: 18px;">GitHub Repository</a>
                    <a href="https://drive.google.com/drive/folders/1gjdmyTR_9B7U1-W7hWugewnSowjetXYC?usp=drive_link" style="color: #5e9ed6; margin: 0 15px; text-decoration: none; font-size: 18px;">Dataset Repository</a>
                </div>
            """)
    
    # Main content
    with gr.Row():
        with gr.Column():
            # Input image section
            input_image = gr.Image(label="Image", type="pil", height=300)
            
        with gr.Column():
            # Language selection dropdown
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div style="color: white; margin-bottom: 8px;">Identified Language</div>')
            
            with gr.Row():
                language = gr.Dropdown(
                    choices=["hindi", "english", "tamil", "telugu", "kannada", "malayalam", "marathi", "gujarati", "punjabi", "bengali", "urdu"],
                    value="hindi",
                    show_label=False
                )
            
            # Buttons
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Submit", variant="primary")
    
    # Set up event handlers
    submit_btn.click(
        fn=process_image,
        inputs=[input_image, language],
        outputs=[]
    )
    
    clear_btn.click(
        fn=lambda: (None, "hindi"),
        inputs=None,
        outputs=[input_image, language]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()