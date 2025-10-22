🤖 Urdu Transformer Chatbot
Advanced Transformer-based Chatbot for Urdu Language with Multi-Response Generation

Demo • Features • Installation • Usage • Architecture • Contributing

📖 Overview
A state-of-the-art Urdu conversational AI chatbot built with Transformer architecture from scratch. Features multiple response generation strategies (Greedy, Beam Search, Temperature Sampling) and a beautiful Gradio interface with proper right-to-left (RTL) text support.

✨ Key Highlights
🧠 Custom Transformer Architecture - 2-layer encoder-decoder with multi-head attention
🎲 Multiple Response Generation - Generate 1-5 diverse responses simultaneously
🎯 Dual Decoding Strategies - Automatic Greedy + Beam Search + Temperature Sampling
🌐 Perfect RTL Support - Noto Nastaliq Urdu font with proper text rendering
⚡ GPU Accelerated - CUDA support for fast inference
🎨 Modern UI - Beautiful gradient-themed Gradio interface
📱 Colab Ready - Deploy instantly on Google Colab with public URL
🎯 Features
🤖 Model Features
Feature	Description
Architecture	Transformer Encoder-Decoder
Parameters	~2-5M (depending on vocab size)
Dimensions	256-dim embeddings, 1024-dim FFN
Attention Heads	2 heads per layer
Layers	2 encoder + 2 decoder layers
Vocab Size	8000 tokens (customizable)
🎲 Generation Strategies
Greedy Decoding ⚡
Fastest generation
Deterministic output
Best for quick responses
Beam Search 🔍
Explores multiple paths (beam size = 3)
Higher quality responses
Better coherence
Temperature Sampling 🎨
Controlled randomness
Creative variations
Adjustable diversity (T = 0.8, 1.0, 1.2, ...)
🎨 Interface Features
✅ Auto-loading Models - No manual loading required
✅ Real-time Chat - Instant responses with conversation history
✅ Response Control - Slider to select 1-5 responses
✅ Sample Questions - Quick-start Urdu questions
✅ Conversation History - Color-coded chat bubbles
✅ Clear Chat - Reset conversation anytime
✅ Model Selection - Choose BLEU or Loss optimized models
🚀 Demo
Live Interface
🤖 اردو چیٹ بوٹ
───────────────────────────────────────
👤 آپ: آپ کیسے ہیں؟

🤖 بوٹ (Greedy):
میں بالکل ٹھیک ہوں، شکریہ! آپ کیسے ہیں؟

🤖 بوٹ (Beam Search):
الحمدللہ، میں اچھا ہوں۔ آپ کا دن کیسا گزر رہا ہے؟

🤖 بوٹ (Sampling T=1.0):
بہت اچھا ہوں! آپ سے مل کر خوشی ہوئی۔
───────────────────────────────────────

📦 Installation
Prerequisites
Python 3.8+
PyTorch 2.0+
CUDA (optional, for GPU acceleration)
Quick Start
bash
# Clone the repository
git clone https://github.com/Umaimtahir/Urdu_Chatbot
cd urdu-chatbot

# Install dependencies
pip install -r requirements.txt

# Run the chatbot
python urdu_chatbot.ipynb

🎮 Usage
Running Locally
python
python urdu_chatbot.ipynb
Access at: http://localhost:7860

Running on Google Colab
python
# 1. Upload your model files
# - best_model_bleu.pt
# - best_model_loss.pt

# 2. Install dependencies
!pip install gradio torch

# 3. Upload and run the script
!python urdu_chatbot.py
Gradio will provide a public URL for sharing!

Using the Interface
Select Model - Choose BLEU or Loss model (auto-loads on first message)
Set Response Count - Use slider to select 1-5 responses
Type Message - Enter Urdu text in the input box
Send - Click button or press Enter
View Responses - See multiple responses with generation methods
Sample Questions
Try these Urdu questions:

آپ کیسے ہیں؟
آج موسم کیسا ہے؟
کیا کر رہے ہیں؟
آپ کا نام کیا ہے؟
مجھے مدد چاہیے
🏗️ Architecture
Model Structure
┌─────────────────────────────────────┐
│         Input Text (Urdu)           │
└────────────┬────────────────────────┘
             │
     ┌───────▼────────┐
     │   Tokenizer    │
     │  (Word → IDs)  │
     └───────┬────────┘
             │
     ┌───────▼────────┐
     │   Embedding    │
     │   (256-dim)    │
     └───────┬────────┘
             │
     ┌───────▼────────┐
     │   Positional   │
     │    Encoding    │
     └───────┬────────┘
             │
   ┌─────────▼─────────┐
   │  ENCODER (2 layers) │
   │  ┌──────────────┐  │
   │  │ Multi-Head   │  │
   │  │  Attention   │  │
   │  └──────┬───────┘  │
   │  ┌──────▼───────┐  │
   │  │ Feed-Forward │  │
   │  └──────────────┘  │
   └─────────┬─────────┘
             │
   ┌─────────▼─────────┐
   │  DECODER (2 layers) │
   │  ┌──────────────┐  │
   │  │ Masked Self- │  │
   │  │  Attention   │  │
   │  └──────┬───────┘  │
   │  ┌──────▼───────┐  │
   │  │ Cross-       │  │
   │  │  Attention   │  │
   │  └──────┬───────┘  │
   │  ┌──────▼───────┐  │
   │  │ Feed-Forward │  │
   │  └──────────────┘  │
   └─────────┬─────────┘
             │
     ┌───────▼────────┐
     │  Output Layer  │
     │  (Vocab Size)  │
     └───────┬────────┘
             │
     ┌───────▼────────┐
     │  Generation    │
     │  Strategy      │
     │ ┌────────────┐ │
     │ │   Greedy   │ │
     │ │ Beam Search│ │
     │ │  Sampling  │ │
     │ └────────────┘ │
     └───────┬────────┘
             │
     ┌───────▼────────┐
     │  Response      │
     │  (Urdu Text)   │
     └────────────────┘
Key Components
1. Multi-Head Attention
python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, heads=2):
        self.w_q = nn.Linear(d_model, d_model)  # Query
        self.w_k = nn.Linear(d_model, d_model)  # Key
        self.w_v = nn.Linear(d_model, d_model)  # Value
        self.w_o = nn.Linear(d_model, d_model)  # Output
Purpose: Allows model to focus on different parts of input simultaneously.

2. Positional Encoding
python
class PositionalEncoding(nn.Module):
    # Sinusoidal position embeddings
    pe[:, 0::2] = sin(position / 10000^(2i/d_model))
    pe[:, 1::2] = cos(position / 10000^(2i/d_model))
Purpose: Adds sequence order information since Transformers have no inherent notion of position.

3. Feed-Forward Network
python
class FeedForward(nn.Module):
    def __init__(self, d_model=256, d_ff=1024):
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
Purpose: Processes each position independently with non-linear transformations.

🎓 Training Your Own Model
Dataset Format
Your training data should be conversation pairs:

json
[
  {"input": "آپ کیسے ہیں؟", "output": "میں ٹھیک ہوں، شکریہ"},
  {"input": "آج موسم کیسا ہے؟", "output": "آج موسم بہت اچھا ہے"},
  ...
]
Training Script (Example)
python
# Pseudocode for training
model = UrduTransformer(vocab_size=8000)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in train_loader:
        src, tgt = batch
        output = model(src, tgt[:, :-1], PAD_ID)
        loss = criterion(output.view(-1, vocab_size), tgt[:, 1:].view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
Model Checkpoints
Save your trained models:

python
torch.save({
    'model_state_dict': model.state_dict(),
    'tokenizer': tokenizer,
    'vocab': vocab,
    'epoch': epoch,
    'loss': loss
}, 'best_model_bleu.pt')
🔧 Configuration
Model Hyperparameters
Edit these in the code:

python
model = UrduTransformer(
    vocab_size=8000,         # Vocabulary size
    d_model=256,             # Model dimension
    heads=2,                 # Attention heads
    num_encoder_layers=2,    # Encoder depth
    num_decoder_layers=2,    # Decoder depth
    d_ff=1024,              # Feed-forward dimension
    max_len=512,            # Max sequence length
    dropout=0.1             # Dropout rate
)
Generation Parameters
python
# Greedy
max_length = 50

# Beam Search
beam_size = 3
max_length = 50

# Temperature Sampling
temperature = 0.8  # Lower = more focused, Higher = more random
max_length = 50
📊 Model Performance
Metrics
Model	BLEU Score	Perplexity	Inference Time
BLEU Optimized	Higher	Moderate	~1.5s (GPU)
Loss Optimized	Moderate	Lower	~1.5s (GPU)
Hardware Requirements
Component	Minimum	Recommended
CPU	4 cores	8+ cores
RAM	8 GB	16+ GB
GPU	-	NVIDIA GTX 1060+
VRAM	-	6+ GB
Storage	2 GB	5+ GB

🐛 Troubleshooting
Common Issues
1. Model Loading Error
Problem: Can't get attribute 'Vocabulary' or 'Vocabulary' object has no attribute 'items'
Solution: The code now includes the Vocabulary class with all required methods. Make sure you're using the latest version of the code.
2. Weights Loading Warning
Problem: Weights only load failed
Solution: The code uses weights_only=False by default. This is safe for trusted model files.
pythoncheckpoint = torch.load(model_path, weights_only=False)
3. CUDA Out of Memory
Problem: RuntimeError: CUDA out of memory
Solution: Reduce model size or use CPU:
pythondevice = torch.device('cpu')
4. Empty Responses
Problem: Model generates empty or very short responses
Solution:

Check your model training
Adjust max_length parameter
Try different generation strategies
Ensure vocabulary is properly loaded

5. File Not Found
Problem: FileNotFoundError: /content/best_model_bleu.pt
Solution: Update paths in code to match your file locations:
pythonmodel_path = '/your/path/to/best_model_bleu.pt'

🌟 Advanced Features
Custom Vocabulary
Add your own vocabulary:
pythontokenizer = SimpleUrduTokenizer()
tokenizer.word2idx = {
    '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3,
    'آپ': 4, 'کیسے': 5, 'ہیں': 6,
    # ... add more words
}
tokenizer.idx2word = {v: k for k, v in tokenizer.word2idx.items()}
tokenizer.vocab_size = len(tokenizer.word2idx)
Response Filtering
Add quality control:
pythondef filter_response(response):
    # Filter empty responses
    if not response or len(response.strip()) < 3:
        return "معذرت، میں جواب نہیں دے سکا"
    
    # Filter repetitive responses
    words = response.split()
    if len(set(words)) < len(words) * 0.5:
        return "معذرت، مناسب جواب نہیں"
    
    return response
Conversation Context
Implement multi-turn conversations:
pythoncontext_window = 3  # Last 3 exchanges
context = []

for user_msg, bot_msg in conversation_history[-context_window:]:
    context.append(user_msg)
    context.append(bot_msg)

# Use context in generation
full_input = " ".join(context + [new_user_input])

📚 API Reference
Model Classes
UrduTransformer
pythonmodel = UrduTransformer(
    vocab_size: int,           # Size of vocabulary
    d_model: int = 256,        # Model dimension
    heads: int = 2,            # Number of attention heads
    num_encoder_layers: int = 2,  # Encoder depth
    num_decoder_layers: int = 2,  # Decoder depth
    d_ff: int = 1024,         # Feed-forward dimension
    max_len: int = 512,       # Maximum sequence length
    dropout: float = 0.1      # Dropout probability
)
SimpleUrduTokenizer
pythontokenizer = SimpleUrduTokenizer()

# Encode text to IDs
ids = tokenizer.encode("آپ کیسے ہیں")  # Returns: [4, 5, 6]

# Decode IDs to text
text = tokenizer.decode([4, 5, 6])     # Returns: "آپ کیسے ہیں"
Generation Functions
greedy_generate
pythonresponse = greedy_generate(
    model: UrduTransformer,
    tokenizer: SimpleUrduTokenizer,
    input_text: str,
    device: torch.device,
    PAD_ID: int,
    BOS_ID: int,
    EOS_ID: int,
    max_length: int = 50
)
beam_search_generate
pythonresponse = beam_search_generate(
    model: UrduTransformer,
    tokenizer: SimpleUrduTokenizer,
    input_text: str,
    device: torch.device,
    PAD_ID: int,
    BOS_ID: int,
    EOS_ID: int,
    beam_size: int = 3,
    max_length: int = 50
)
sampling_generate
pythonresponse = sampling_generate(
    model: UrduTransformer,
    tokenizer: SimpleUrduTokenizer,
    input_text: str,
    device: torch.device,
    PAD_ID: int,
    BOS_ID: int,
    EOS_ID: int,
    temperature: float = 0.8,
    max_length: int = 50
)

🤝 Contributing
We welcome contributions! Here's how:
Reporting Bugs

Check existing issues
Create detailed bug report with:

Error message
Steps to reproduce
Environment details



Proposing Features

Open an issue with [Feature Request] tag
Describe the feature and use case
Discuss implementation approach

Code Contributions
bash# Fork the repository
git clone https://github.com/Umaimtahir/Urdu_Chatbot
cd urdu-chatbot


📊 Datasets: Add more Urdu conversation data
🎨 UI/UX: Improve interface design
🧠 Models: Implement new architectures (GPT, BERT variants)
🔧 Features: Add voice support, emotion detection
📖 Documentation: Improve guides and examples
🌍 Localization: Add more languages


📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
MIT License

Copyright (c) 2024 Umaim Tahir

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...

🙏 Acknowledgments

PyTorch Team - For the deep learning framework
Gradio Team - For the amazing UI library
Hugging Face - For Transformers inspiration
Urdu NLP Community - For datasets and resources
Contributors - Everyone who helps improve this project

Built With

PyTorch - Deep learning framework
Gradio - Web UI framework
NumPy - Numerical computing


📞 Contact & Support

GitHub Issues: Report bugs or request features
Email: umaimtahir1@gmail.com


🗺️ Roadmap
Version 1.0 (Current)

✅ Basic Transformer architecture
✅ Greedy and Beam Search
✅ Gradio interface
✅ Multi-response generation

Version 1.5 (Planned)

🔲 Attention visualization
🔲 Voice input/output
🔲 Conversation memory
🔲 Response ranking

Version 2.0 (Future)

🔲 Fine-tuning interface
🔲 Multi-language support
🔲 API endpoint
🔲 Mobile app


Solution: Make sure Vocabulary

