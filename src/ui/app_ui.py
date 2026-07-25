# src/ui/app_ui.py

import gradio as gr


def create_ui(respond, upload_pdf):

    with gr.Blocks(
        title="ScholarRAG",
        theme=gr.themes.Soft()
    ) as demo:

        # ==================================================
        # Header
        # ==================================================

        gr.Markdown(
            """
            # 🎓 ScholarRAG

            ### AI-Powered Research Paper Explainer

            Upload a research paper and ask anything about it.
            """
        )

        # ==================================================
        # Upload Section
        # ==================================================

        with gr.Group():

            gr.Markdown("## 📄 Research Paper")

            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="filepath"
            )

            upload_button = gr.Button(
                "Upload & Process Paper",
                variant="primary"
            )

            upload_status = gr.Markdown(
                "No research paper uploaded."
            )

        # ==================================================
        # Research Assistant
        # ==================================================

        gr.Markdown("## 💬 Ask ScholarRAG")

        chatbot = gr.Chatbot(
            label="Research Assistant",
            height=450,
            type="tuples"
        )

        message = gr.Textbox(
            label="Your Question",
            placeholder=(
                "Ask anything about the uploaded paper..."
            ),
            lines=2
        )

        with gr.Row():

            send_button = gr.Button(
                "Send",
                variant="primary"
            )

            clear_button = gr.Button(
                "Clear Conversation"
            )

        # ==================================================
        # Events
        # ==================================================

        upload_button.click(
            fn=upload_pdf,
            inputs=pdf_input,
            outputs=upload_status
        )

        send_button.click(
            fn=respond,
            inputs=[
                message,
                chatbot
            ],
            outputs=chatbot
        )

        message.submit(
            fn=respond,
            inputs=[
                message,
                chatbot
            ],
            outputs=chatbot
        )

        clear_button.click(
            fn=lambda: [],
            outputs=chatbot,
            queue=False
        )

    return demo