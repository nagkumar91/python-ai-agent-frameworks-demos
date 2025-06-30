import asyncio

import httpx


async def fetch_url(url):
    print(f"Fetching {url}")
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        r = await client.get(url)
        print(f"Fetched {url} with status code {r.status_code}")
        return r.text


async def main():
    async with asyncio.TaskGroup() as tg:
        tg.create_task(fetch_url("https://httpbin.org/delay/3"))
        tg.create_task(fetch_url("https://httpbin.org/delay/1"))


asyncio.run(main())
