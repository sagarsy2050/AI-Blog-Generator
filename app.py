import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Streamlit UI Setup
st.set_page_config(page_title="AI Blog Generator", page_icon="📝", layout="centered")

st.title("📝 AI Blog Generator")
st.write("Generate engaging blog posts effortlessly using AI!")

# User Inputs
input_text = st.text_input("Enter the blog topic:", placeholder="e.g., The Future of AI")
no_words = st.slider("Select the desired word count:", min_value=100, max_value=1000, step=50, value=300)
blog_style = st.selectbox("Choose a blog style:", ["Informative", "Persuasive", "Casual", "Professional"])


# Function to generate a blog
def generate_blog(topic, word_count, style):
    # Define prompt template
    prompt_template = """
    Write a {style} blog post on the topic: {topic}.
    Word Count: Approximately {word_count} words.
    Ensure it is engaging and informative.
    """         

    # Format the prompt
    prompt = PromptTemplate(template=prompt_template, input_variables=["topic", "word_count", "style"])
    formatted_prompt = prompt.format(topic=topic, word_count=word_count, style=style)

    # Initialize Llama model correctly
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGUF",
        model_type="llama",
        config={
            "temperature": 0.7,  # Control creativity
            "max_new_tokens": word_count  # Control word count
        }
    )

    # Generate the blog
    return llm.invoke(formatted_prompt)
# # Function to generate a blog
# def generate_blog(topic, word_count, style):
#     # Define prompt template
#     prompt_template = """
#     Write a {style} blog post on the topic: {topic}.
#     Word Count: Approximately {word_count} words.
#     Ensure it is engaging and informative.
#     """

#     # Format the prompt
#     prompt = PromptTemplate(template=prompt_template, input_variables=["topic", "word_count", "style"])
#     formatted_prompt = prompt.format(topic=topic, word_count=word_count, style=style)

#     llm = CTransformers(
#         model="irresistiblegrace97/Llama-2-7b-chat-hf-Q2_K-GGUF",  #TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q2_K.gguf
#         model_type="llama",
#         config={"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": word_count, "temperature": 0.7}
#     )

#     # Generate the blog
#     return llm.invoke(formatted_prompt)

# Generate Button
if st.button("Generate Blog"):
    if not input_text.strip():
        st.warning("⚠️ Please enter a blog topic.")
    else:
        with st.spinner("⏳ Generating your blog... Please wait."):
            blog_content = generate_blog(input_text, no_words, blog_style)
            st.success("✅ Blog generated successfully!")
            st.text_area("Generated Blog", value=blog_content, height=400)

# Footer
st.markdown("---")
st.caption("🚀 Powered by LangChain & LLaMA Models")
