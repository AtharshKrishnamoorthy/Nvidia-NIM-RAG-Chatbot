# 🤖 RAG Chatbot using Nvidia NIM

Ever wanted to chat with your PDF documents? Well, now you can! This project brings your PDFs to life using the power of NVIDIA NIM (Nvidia Inference Microservices) and Retrieval-Augmented Generation (RAG). Simply upload your PDFs and start asking questions - it's like having a smart study buddy who's read all your documents! 

## 📸 Screenshots

Here's a peek at what the chatbot looks like in action:

![{C306FB9B-692A-4F7D-90CA-99317AEC3B27}](https://github.com/user-attachments/assets/974faa56-e9ac-4bff-a0b9-8e2c50621878)

![{D3F02B05-F041-4ECC-8B33-74E5027C8FB1}](https://github.com/user-attachments/assets/b62bfa5f-e961-484b-8896-7462fb18a8c5)



## ✨ What Makes This Special?

- 📚 **Smart Document Understanding**: Upload any PDF and watch as the chatbot understands and learns from it
- 🧠 **Contextual Memory**: The bot remembers your conversation, making each interaction more meaningful
- 🎯 **Precise Answers**: Gets information directly from your documents, no hallucinations!
- 🚀 **Powered by Nvidia**: Leverages NIM for accessing the open source LLM models using the NVIDIA API KEY

## 🛠️ Getting Started

### Prerequisites

First, make sure you have:
- Python 3.7 or newer installed
- A Nvidia API key (your ticket to AI magic!)
- Some PDFs you want to chat about

### Quick Setup

1. **Clone the repo:**
```bash
git clone https://github.com/AtharshKrishnamoorthy/Nvidia-NIM-RAG-Chatbot
cd Nvidia-NIM-RAG-Chatbot
```

2. **Install the magic ingredients:**
```bash
pip install -r requirements.txt
```

3. **Set up your environment:**
```bash
cp .env.template .env
```
Then edit `.env` and add your Nvidia API key. Keep this secret, keep it safe! 🧙‍♂️

### 🚀 Launch Time!

Fire up the chatbot:
```bash
streamlit run app2.py
```

## 🎯 How to Use

1. **Upload Your PDFs:**
   - Head to the sidebar
   - Click upload and choose your PDFs
   - Grab a coffee while the bot processes them ☕

2. **Start Chatting:**
   - Type your questions in the chat box
   - Watch as the bot finds relevant info from your PDFs
   - Get clear, contextual answers

3. **Need a Fresh Start?**
   - Hit the "Clear Chat History" button
   - Begin a new conversation!

## 🔧 Under the Hood

This chatbot is built with some seriously cool tech:
- **Streamlit**: For that sleek, user-friendly interface
- **LangChain**: Provides the orchestration framework for the chatbot built. 
- **FAISS**: Local vector store of storing the Embeddings
- **Nvidia NIM**: Using NIM -> utilized the llama-3.1-nemotron-70b-instruct model by using the API key

## 🤝 Want to Contribute?

We love contributions! Here's how you can help:

1. Fork the repo
2. Create your feature branch: `git checkout -b cool-new-feature`
3. Commit your changes: `git commit -m 'Added something awesome'`
4. Push to the branch: `git push origin cool-new-feature`
5. Open a Pull Request

## 📝 Environment Setup

Create a `.env` file with these variables:
```
NVIDIA_API_KEY=your_api_key_here
```

Don't worry, it's already in `.gitignore` to keep your secrets safe!

## 📜 License

This project is under the MIT License - because sharing is caring! See [LICENSE.md](LICENSE) for the legal stuff.

## Contact

If you have any questions, feel free to reach out:

- Project Maintainer: Atharsh K
- Email: atharshkrishnamoorthy@gmail.com
- Project Link: [GitHub Repository](https://github.com/AtharshKrishnamoorthy/Nvidia-NIM-RAG-Chatbot)

