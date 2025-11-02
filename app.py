import streamlit as st
import os

st.set_page_config(page_title="AI Chatbot with Prompt Controls", layout="wide")

MODEL_CHOICES = {
    "Llama-3.1-8b-Instant (Groq)": "llama-3.1-8b-instant",
    "Llama-3.3-70b-Versatile (Groq)": "llama-3.3-70b-versatile"
}

try:
    from groq import Groq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    if hasattr(st, "secrets"):
        GROQ_API_KEY = GROQ_API_KEY or st.secrets.get("GROQ_API_KEY", "")
    client = Groq(api_key=GROQ_API_KEY)
except ImportError:
    client = None
    st.warning("Groq client not found. Install with: pip install groq")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

st.sidebar.title("Model & Settings")

model_name = st.sidebar.selectbox("Select Model", list(MODEL_CHOICES.keys()), index=0)
model_id = MODEL_CHOICES[model_name]

with st.sidebar.expander("Advanced Settings"):
    temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.05)
    max_tokens = st.number_input("Max Tokens", 256, 8192, 1024, 128)
    top_p = st.slider("Top P", 0.0, 1.0, 1.0, 0.05)

st.sidebar.markdown("---")
st.sidebar.title("Prompt Engineering")

persona_presets = {
    "Custom": "",
    "Helpful Assistant": "You are a helpful, knowledgeable AI assistant.",
    "Code Expert": "You are an expert software engineer and programmer.",
    "Data Scientist": "You are a data scientist with expertise in ML and statistics.",
    "Academic Tutor": "You are a knowledgeable academic tutor who explains concepts thoroughly.",
    "Creative Writer": "You are a creative writer with strong storytelling abilities.",
    "Business Consultant": "You are a professional business consultant with industry expertise."
}

selected_persona = st.sidebar.selectbox("Persona", list(persona_presets.keys()), index=0)
persona = persona_presets[selected_persona] if selected_persona != "Custom" else st.sidebar.text_input("Custom Persona", value="")

context_presets = {
    "Custom": "",
    "Explanation": "Explain concepts clearly and concisely with examples.",
    "Code Generation": "Generate clean, well-documented code with comments.",
    "Problem Solving": "Help solve problems step-by-step with clear reasoning.",
    "Brainstorming": "Generate creative ideas and discuss possibilities openly.",
    "Learning": "Teach and explain topics in an educational manner.",
    "Analysis": "Analyze information critically and provide insights."
}

selected_context = st.sidebar.selectbox("Context/Task", list(context_presets.keys()), index=0)
task = context_presets[selected_context] if selected_context != "Custom" else st.sidebar.text_area("Custom Task", value="", height=60)

format_presets = {
    "Markdown": "Markdown",
    "Bullet Points": "Bullet Points",
    "Code Blocks": "Code Blocks with syntax highlighting",
    "Step-by-Step": "Numbered steps",
    "JSON": "JSON format",
    "Table": "Table format",
    "Custom": ""
}

selected_format = st.sidebar.selectbox("Format Style", list(format_presets.keys()), index=0)
format_style = format_presets[selected_format] if selected_format != "Custom" else st.sidebar.text_input("Custom Format", value="")

tone = st.sidebar.selectbox("Tone", ["Professional", "Casual", "Friendly", "Technical", "Academic", "Humorous", "Formal"], index=0)

with st.sidebar.expander("Examples (Optional)"):
    examples = st.text_area("Input/Output Examples", value="", height=100, placeholder="E.g., Input: 'What is ML?'\nOutput: 'ML is...'")

def build_system_prompt():
    parts = []
    if persona:
        parts.append(persona)
    if task:
        parts.append(f"Task: {task}")
    if examples:
        parts.append(f"Examples:\n{examples}")
    if format_style:
        parts.append(f"Format: {format_style}")
    if tone:
        parts.append(f"Tone: {tone}")
    return "\n".join(parts)

with st.sidebar.expander("Current Prompt Config"):
    st.markdown("### System Prompt Preview:")
    st.code(build_system_prompt(), language="text")

if st.sidebar.button("Summarize Chat"):
    if client:
        conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
        prompt = f"Summarize the following conversation:\n{conversation}"
        with st.spinner("Summarizing..."):
            summary = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            ).choices[0].message.content
        st.sidebar.markdown("### Conversation Summary")
        st.sidebar.info(summary)
    else:
        st.sidebar.warning("Groq API not connected.")

st.title("AI Chatbot")
st.caption(f"Model: {model_name}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    system_prompt = build_system_prompt()
    
    with st.chat_message("assistant"):
        if client:
            with st.spinner("Thinking..."):
                messages_to_send = [{"role": "system", "content": system_prompt}]
                messages_to_send.extend(st.session_state.messages)
                
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages_to_send,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                ).choices[0].message.content
                
                st.markdown(response)
        else:
            response = "Waiting for API connection..."
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

st.sidebar.markdown("---")
st.sidebar.caption("Deploy to Hugging Face Spaces. Add GROQ_API_KEY as a secret.")
