---
title: 6.C395 VoteMatch
emoji: ✅ 
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.23.3
python_version: "3.10"
# app_file: app.py
pinned: false
secrets:
  - HF_TOKEN
---

# 6.C395 VoteMatch

## Setup

1. Make a virtual environment and install the required dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Make a HuggingFace account and make an access token:
   - Visit [Hugging Face](https://huggingface.co)
   - Make an account if you don't already have one
   - Click on your profile, then "Access Tokens" and make a new token
   - Make a .env file with `HF_TOKEN=<insert your token here>`
   - Now, log in to Hugging Face in the terminal as well:
   ```bash
   huggingface-cli login
   ```

3. Choose a base model:
   - In config.py, set the BASE_MODEL variable to your base model of choice from HuggingFace.
   - Keep in mind it's better to have a small, lightweight model if you plan on finetuning.


## Deploying to Hugging Face

To deploy as a free web interface using Hugging Face Spaces:

1. Create a Hugging Face Space:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "New Space"
   - Choose a name for your space (e.g., "6.C395-chatbot")
   - Select "Gradio" as the SDK
   - Choose "CPU" as the hardware (free tier)
   - Make it "Public" so others can use your chatbot

2. Prepare your files:
   Your repository should already have all needed files:
   ```
   6.c395-chatbot/
   ├── README.md           # Description of your chatbot
   ├── app.py             # Your Gradio interface
   ├── requirements.txt   # Already set up with needed dependencies
   └── src/              # Your implementation files
   ```

3. Push your code to the Space:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push -u origin main
   ```

4. Add your HF_TOKEN to the space as a secret.
   - Go to Files.
   - Go to Settings.
   - Under secrets, add HF_TOKEN.
   

5. Important Free Tier Considerations:
   - The default model (meta-llama/Llama-3.1-8B-Instruct) runs via HuggingFace's Inference Providers, not on your Space's CPU. Your Space just hosts the Gradio UI.
   - Free HuggingFace accounts have a limited monthly credit quota for Inference Providers. You may hit a 402 "Payment Required" error if you exceed it. To conserve credits, test locally when possible (`python app.py`) and avoid unnecessary requests.
   - The interface might queue requests when multiple users access it. Sometimes there will be 503 errors. Just try again a few seconds later.

6. After Deployment:
   - Your chatbot will be available at: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`
   - Anyone can use it through their web browser
   - You can update the deployment anytime by pushing changes:
     ```bash
     git add .
     git commit -m "Update chatbot"
     git push
     ```

7. Troubleshooting:
   - Check the Space's logs if the chatbot isn't working
   - Verify the chatbot works locally before deploying
   - Remember free tier has limited resources. Sometimes if you get a 503 error it means the server is overloaded. Just try again a few seconds later.

