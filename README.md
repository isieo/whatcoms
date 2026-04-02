# Whatcoms: Daily Communications Summarizer

A powerful, privacy-first WhatsApp bot that uses high-performance local Vision-Language Models (VLMs) to summarize your chats. It can read messages, understand context, and even analyze images sent in your groups or private chats.

---

## 🚀 Key Features

-   **Privacy First:** Everything runs locally on your machine. No chat data is sent to external AI APIs.
-   **Multimodal Vision:** Uses **Qwen2-VL-7B** to actually *see* and describe images sent in your chats.
-   **VRAM Management:** Dynamically loads the model into VRAM/GPU only when a request is made and immediately disposes of it afterwards to keep your system resources free when idle.
-   **Smart Caching:** Incremental chat caching with automatic daily rotation to keep the bot fast and your disk clean.
-   **Auto-Downloader:** Automatically detects, downloads, and initializes the required GGUF models upon first run.
-   **Flexible Commands:**
    -   `!summary`: Get a comprehensive summary of today's key discussions and action items.
    -   `!hourly`: Get a quick recap of the last 60 minutes.
    -   `!ask <question>`: Ask a specific question about today's chat history.

---

## 🏗️ Prerequisites

-   **Node.js**: Version 18.20+ or 20+ (required for `node-llama-cpp`).
-   **Hardware**: A GPU (NVIDIA recommended for CUDA speed) is highly recommended, though it can run on a powerful CPU.
-   **WhatsApp**: A phone with WhatsApp to scan the QR code for authentication.

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone git@github.com:isieo/whatcoms.git
   cd WhatsappSummarizer
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Set up Environment Variables:**
   Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and enter your WhatsApp ID (usually your phone number + country code followed by `@c.us`):
   ```env
   WHATSAPP_OWNER_ID=60123456789@c.us
   ```

---

## 🏃 Usage

1. **Start the bot:**
   ```bash
   node index.js
   ```

2. **Authenticate:**
   Scan the QR code that appears in your terminal with your WhatsApp mobile app (Linked Devices).

3. **Commands (Owner Only):**
   Send these commands from your phone to any chat or directly to the bot:
   -   `!summary [instructions]`: Summarize the whole day.
   -   `!hourly [instructions]`: Summarize the last hour.
   -   `!ask <your question>`: Query the chat logs for specific info.

---

## 🤝 Support

This project uses local GGUF models. If you have any issues with model performance, ensure your GPU drivers are updated and CUDA is installed properly for `node-llama-cpp`.

---

✨ This project is **vibecoded**.
