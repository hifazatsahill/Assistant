import os
import chainlit as cl
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
from dotenv import load_dotenv
import asyncio

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# External client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Run configuration
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# ✅ FIX: use `mcp_config` instead of `run_config`
agent1 = Agent(
    name="agent1",
    instructions="You are specialized for General query",
    model=model
)
agent2 = Agent(
    name="agent2",
    instructions="You are specialized for math query",
    model=model
)
agent3 = Agent(
    name="agent3",
    instructions="You are specialized for physics questions and answers",
    model=model
)
agent4 = Agent(
    name="agent4",
    instructions="You are specialized for English questions and answers. Grammar.",
    model=model
)
agent5 = Agent(
    name="agent5",
    instructions="You are specialized for chemistry questions and answers.",
    model=model
)
main_agent = Agent(
    name="assistant",
    instructions="You are specialized for delegation agent",
    model=model,
    handoffs=[agent1, agent2, agent3, agent4, agent5]
)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Welcome to the AI Assistant").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    history.append({"role": "user", "content": message.content})

    # Runner call
    result = await Runner.run(
        main_agent,
        input=history,
        run_config=config
    )

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)

    await cl.Message(content=result.final_output).send()
