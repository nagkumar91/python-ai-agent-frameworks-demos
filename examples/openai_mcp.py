import asyncio
import logging
import os

import azure.identity
import openai
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from rich.logging import RichHandler

# Setup logging with rich
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("weekend_planner")

# Disable tracing since we're not connected to a supported tracing provider
set_tracing_disabled(disabled=True)

# Setup the OpenAI client to use either Azure OpenAI or GitHub Models
load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")
if API_HOST == "github":
    client = openai.AsyncOpenAI(base_url="https://models.inference.ai.azure.com", api_key=os.environ["GITHUB_TOKEN"])
    MODEL_NAME = os.getenv("GITHUB_MODEL", "gpt-4o")
elif API_HOST == "azure":
    token_provider = azure.identity.get_bearer_token_provider(azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    client = openai.AsyncAzureOpenAI(
        api_version=os.environ["AZURE_OPENAI_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,
    )
    MODEL_NAME = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
elif API_HOST == "ollama":
    client = openai.AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="none")
    MODEL_NAME = "llama3.1:latest"

NUM_TO_ACCEPT = 100


async def process_linkedin_invitations():
    """
    Processes LinkedIn invitations based on criteria:
    - Ignore recruiters
    - Accept people with technical roles who have mutual connections
    - Process up to 10 invitations
    """
    print("Starting LinkedIn invitation processing...")
    results = []
    processed_count = 0

    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=False)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()

        # Create a new page
        page = await context.new_page()

        # Navigate to LinkedIn login page
        await page.goto("https://www.linkedin.com/login")
        print("Please log in to LinkedIn manually...")

        # Wait for login to complete (wait for feed page)
        await page.wait_for_url("https://www.linkedin.com/feed/**", timeout=120000)
        print("Login detected. Navigating to invitation manager...")

        # Go to invitation manager
        await page.goto("https://www.linkedin.com/mynetwork/invitation-manager/")
        await page.wait_for_load_state("load")

        # Process invitations using the componentkey selectors from the screenshot
        await page.wait_for_selector("[componentkey='InvitationManagerPage_InvitationsList']")
        invitation_cards = await page.query_selector_all("[componentkey='InvitationManagerPage_InvitationsList'] > div[componentkey^='auto-component-']")
        print(f"Found {len(invitation_cards)} initial invitation cards")

        # Initialize the last processed index
        last_processed_index = 0

        while processed_count < NUM_TO_ACCEPT and len(invitation_cards) > 0:
            for index, card in enumerate(invitation_cards):
                # Skip cards that have already been processed
                if index < last_processed_index:
                    continue

                if processed_count >= NUM_TO_ACCEPT:
                    break

                # Extract profile info
                name_element = await card.query_selector("a > strong")
                if not name_element:
                    continue
                name = await name_element.inner_text()

                # Get profile link
                profile_link_element = await card.query_selector("a")
                profile_link = await profile_link_element.get_attribute("href") if profile_link_element else "Unknown"
                if not profile_link.startswith("http"):
                    profile_link = f"https://www.linkedin.com{profile_link}"

                # Get headline/title
                headline_element = await card.query_selector("p:has-text('inviting you to connect')")
                headline = await headline_element.inner_text() if headline_element else "Unknown"

                # Get job title
                job_title_element = await card.query_selector("p:nth-of-type(2)")
                job_title = await job_title_element.inner_text() if job_title_element else "Unknown"

                # Check for mutual connections
                connection_info_element = await card.query_selector("*:has-text('mutual connection')")
                connection_info = await connection_info_element.inner_text() if connection_info_element else ""
                has_mutual_connections = "mutual connection" in connection_info.lower()

                # Decision logic
                is_recruiter = any(term in job_title.lower() for term in ["recruit", "talent", "hr", "sourcing"])
                is_technical = any(term in job_title.lower() for term in ["engineer", "developer", "programmer", "software", "data", "tech", "code"])

                decision = "undecided"
                if is_recruiter:
                    decision = "ignore"
                elif is_technical and has_mutual_connections:
                    decision = "accept"

                # Process the invitation based on decision
                if decision == "accept":
                    accept_button = await card.query_selector("button[aria-label*='Accept']")
                    if accept_button:
                        await accept_button.click()
                        print(f"Accepted invitation from {name} ({headline})")
                elif decision == "ignore":
                    ignore_button = await card.query_selector("button[aria-label*='Ignore']")
                    if ignore_button:
                        await ignore_button.click()
                        print(f"Ignored invitation from {name} ({headline})")

                # Store results
                results.append({"name": name, "profile": profile_link, "headline": headline, "job_title": job_title, "mutual_connections": has_mutual_connections, "is_recruiter": is_recruiter, "is_technical": is_technical, "decision": decision})

                processed_count += 1

                # Small delay to avoid rate limiting
                await asyncio.sleep(1)

                if processed_count >= NUM_TO_ACCEPT:
                    break

            # Update the last processed index
            last_processed_index = len(invitation_cards)

            if processed_count < NUM_TO_ACCEPT:
                # Scroll down to load more
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(2)  # Wait for new cards to load

                # Get updated cards
                invitation_cards = await page.query_selector_all("[componentkey='InvitationManagerPage_InvitationsList'] > div[componentkey^='auto-component-']")

                if len(invitation_cards) == 0:
                    print("No more invitations found")
                    break

        # Generate report
        print("\n=== LinkedIn Invitation Processing Report ===")
        print(f"Total invitations processed: {processed_count}")

        accepted = [r for r in results if r["decision"] == "accept"]
        ignored = [r for r in results if r["decision"] == "ignore"]
        undecided = [r for r in results if r["decision"] == "undecided"]

        print(f"\nAccepted ({len(accepted)}):")
        for invitation in accepted:
            print(f"- {invitation['name']} ({invitation['headline']})")
            print(f"  Profile: {invitation['profile']}")
            print(f"  Mutual connections: {invitation['mutual_connections']}")

        print(f"\nIgnored ({len(ignored)}):")
        for invitation in ignored:
            print(f"- {invitation['name']} ({invitation['headline']})")
            print(f"  Profile: {invitation['profile']}")

        print(f"\nUndecided ({len(undecided)}):")
        for invitation in undecided:
            print(f"- {invitation['name']} ({invitation['headline']})")
            print(f"  Profile: {invitation['profile']}")

        # Close the browser
        await browser.close()

    return results


async def run():
    agent = Agent(
        name="Assistant",
        instructions="Use the Playwright tool to interact with the LinkedIn website.",
        model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    )

    message = "Process the invitation and decide whether to accept or ignore it. If you accept, provide a link to the profile."
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)


if __name__ == "__main__":
    # Comment out the original function and call the LinkedIn invitation processor
    # asyncio.run(run())
    asyncio.run(process_linkedin_invitations())
