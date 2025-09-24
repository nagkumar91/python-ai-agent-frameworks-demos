import os

import azure.identity
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_azure_ai.callbacks.tracers import AzureAIInferenceTracer
from rich import print

load_dotenv(override=True)

azure_tracer = AzureAIInferenceTracer(
    connection_string=os.environ.get("APPLICATION_INSIGHTS_CONNECTION_STRING"),
    enable_content_recording=True,
    name="Basic informational agent",
    id="basic_informational_agent_001",
    endpoint="https://models.inference.ai.azure.com",
    scope="General information"
)
tracers = [
    azure_tracer
]

API_HOST = os.getenv("API_HOST", "github")

if API_HOST == "azure":
    token_provider = azure.identity.get_bearer_token_provider(
        azure.identity.DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    model = AzureChatOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        openai_api_version=os.environ.get("AZURE_OPENAI_VERSION"),
        azure_ad_token_provider=token_provider,
    )
elif API_HOST == "github":
    model = ChatOpenAI(
        model=os.getenv("GITHUB_MODEL", "gpt-4o"),
        base_url="https://models.inference.ai.azure.com",
        api_key=os.environ.get("GITHUB_TOKEN"),
    )
elif API_HOST == "ollama":
    model = ChatOpenAI(
        model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
        base_url=os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1"),
        api_key="none",
    )
else:
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

agent = create_agent(model=model, prompt="You're an informational agent. Answer questions cheerfully.", tools=[])


def main():
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Whats weather today in San Francisco?"}]},
        config={"callbacks": [azure_tracer]}
    )
    latest_message = response["messages"][-1]
    print(latest_message.content)


if __name__ == "__main__":
    main()
