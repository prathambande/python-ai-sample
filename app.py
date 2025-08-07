import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_openai import AzureChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
import asyncio
import json
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AccessToken

app = FastAPI()

endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")

# Use Managed Identity to get a token for Azure OpenAI
credential = DefaultAzureCredential()
token = credential.get_token("https://cognitiveservices.azure.com/.default")


# LLM for long answer (detailed)
llm_long = AzureChatOpenAI(
    azure_endpoint=endpoint,
    openai_api_version="2025-01-01-preview",
    deployment_name=deployment,
    temperature=0.5,
    streaming=True,
    max_tokens=600,
    azure_ad_token=token.token
)

# LLM for summary (shorter)
llm_summary = AzureChatOpenAI(
    azure_endpoint=endpoint,
    openai_api_version="2025-01-01-preview",
    deployment_name=deployment,
    temperature=0,
    max_tokens=200,
    azure_ad_token=token.token
)

summarize_chain = load_summarize_chain(llm_summary, chain_type="stuff")

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ask & Summarize</title>
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6fb; margin: 0; }
            #chat-container { max-width: 700px; margin: 40px auto; background: #fff; border-radius: 10px; box-shadow: 0 2px 8px #0001; padding: 24px; }
            #chat { min-height: 300px; }
            .msg { display: flex; margin: 12px 0; }
            .bubble {
                padding: 12px 18px;
                border-radius: 18px;
                max-width: 80%;
                word-break: break-word;
                font-size: 1.05em;
                box-shadow: 0 1px 3px #0001;
            }
            .user { justify-content: flex-end; }
            .user .bubble { background: #0078d4; color: #fff; border-bottom-right-radius: 4px; }
            .bot { justify-content: flex-start; }
            .bot .bubble { background: #e5eaf1; color: #222; border-bottom-left-radius: 4px; }
            .summary { background: #fffbe6; color: #7c6f00; border: 1px solid #ffe58f; margin-top: 8px; }
            #chat-form { display: flex; margin-top: 18px; }
            #user-input { flex: 1; padding: 10px; font-size: 1em; border-radius: 6px; border: 1px solid #ccc; }
            #send-btn { margin-left: 8px; padding: 10px 20px; font-size: 1em; border-radius: 6px; border: none; background: #0078d4; color: #fff; cursor: pointer; }
            #send-btn:disabled { background: #aaa; }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    </head>
    <body>
        <div id="chat-container">
            <div id="chat"></div>
            <form id="chat-form">
                <input type="text" id="user-input" autocomplete="off" placeholder="Ask a question..." required />
                <button id="send-btn" type="submit">Send</button>
            </form>
        </div>
        <script>
            const chat = document.getElementById('chat');
            const form = document.getElementById('chat-form');
            const input = document.getElementById('user-input');
            const sendBtn = document.getElementById('send-btn');

            function appendMessage(role, html) {
                const msgDiv = document.createElement('div');
                msgDiv.className = 'msg ' + role;
                const bubble = document.createElement('div');
                bubble.className = 'bubble';
                bubble.innerHTML = html;
                msgDiv.appendChild(bubble);
                chat.appendChild(msgDiv);
                chat.scrollTop = chat.scrollHeight;
                return bubble;
            }

            form.onsubmit = async (e) => {
                e.preventDefault();
                const userMsg = input.value;
                appendMessage('user', userMsg);
                input.value = '';
                sendBtn.disabled = true;

                // Show loading message
                let botMsgDiv = document.createElement('div');
                botMsgDiv.className = 'msg bot';
                let bubble = document.createElement('div');
                bubble.className = 'bubble';
                bubble.innerHTML = "<b>Long answer:</b><br><i>Generating...</i>";
                botMsgDiv.appendChild(bubble);
                chat.appendChild(botMsgDiv);
                chat.scrollTop = chat.scrollHeight;

                // Streaming fetch for long answer and then summary
                let longText = '';
                const res = await fetch('/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: userMsg})
                });

                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let done = false;
                let summaryShown = false;
                while (!done) {
                    const { value, done: doneReading } = await reader.read();
                    done = doneReading;
                    if (value) {
                        const chunk = decoder.decode(value, {stream: true});
                        // Check for summary marker
                        if (chunk.startsWith("__SUMMARY__")) {
                            // Show summary
                            const summary = chunk.replace("__SUMMARY__", "");
                            let summaryDiv = document.createElement('div');
                            summaryDiv.className = 'bubble summary';
                            summaryDiv.innerHTML = "<b>Summary:</b><br>" + marked.parse(summary);
                            botMsgDiv.appendChild(summaryDiv);
                            summaryShown = true;
                        } else {
                            longText += chunk;
                            bubble.innerHTML = "<b>Long answer:</b><br>" + marked.parse(longText);
                            chat.scrollTop = chat.scrollHeight;
                        }
                    }
                }
                sendBtn.disabled = false;
            };
        </script>
    </body>
    </html>
    """

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "")

    messages = [
        SystemMessage(content="You are an AI assistant. Please provide a detailed, comprehensive answer to the user's question."),
        HumanMessage(content=question)
    ]

    async def streamer():
        # 1. Stream the long answer
        long_answer = ""
        loop = asyncio.get_event_loop()
        for chunk in llm_long.stream(messages):
            long_answer += chunk.content
            yield chunk.content
            await asyncio.sleep(0)  # Yield control to event loop

        # 2. Summarize after long answer is complete
        docs = [Document(page_content=long_answer)]
        print("Generating summary...")  # Add this
        summary = await loop.run_in_executor(None, summarize_chain.run, docs)
        print("Summary generated:", summary)  # Add this
        yield "__SUMMARY__" + summary

    return StreamingResponse(streamer(), media_type="text/plain")