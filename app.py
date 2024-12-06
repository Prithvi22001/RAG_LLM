from utils import generate_runbook
import gradio as gr


# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 📝 Databricks on AWS Runbook Generator")

    # Input fields for query
    query = gr.Textbox(
        type="text", 
        label="Query", 
        placeholder="Enter your query, e.g., 'Set up Databricks on AWS?'"
    )

    # Output area for the generated runbook as Markdown
    runbook_output = gr.Markdown(label="Generated Databricks Setup Runbook")

    # Submit button to generate the runbook
    submit_button = gr.Button("Generate Runbook", variant="primary")

    # Button action to trigger the RAG pipeline
    submit_button.click(
        fn=generate_runbook,  # Call the synchronous wrapper function
        inputs=query,         # Pass the input query
        outputs=runbook_output  # Display the output as Markdown
    )

# Launch the Gradio app
demo.launch()
