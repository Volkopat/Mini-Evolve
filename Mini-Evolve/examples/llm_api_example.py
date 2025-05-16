import yaml
# from openai import OpenAI # Using httpx now
import httpx # Added for async
import asyncio # Added for async

# Example: using google-generativeai (ensure you have the 'google-generativeai' package installed)
# import google.generativeai as genai

# Load configuration
CONFIG_FILE = "config/config.yaml"

def load_config():
    try:
        with open(CONFIG_FILE, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: Configuration file %s not found." % CONFIG_FILE)
        return None
    except yaml.YAMLError as e:
        print("Error parsing YAML configuration: %s" % e)
        return None

config = load_config()

async def get_llm_response(prompt_text: str, client: httpx.AsyncClient) -> str | None:
    if not config or 'llm' not in config:
        print("LLM configuration section is missing.")
        return None

    llm_config = config['llm']
    api_key = llm_config.get('api_key')
    model_name = llm_config.get('model_name')
    provider = llm_config.get('provider')
    base_url = llm_config.get('base_url')

    if not provider or not model_name:
        print("LLM provider or model_name is missing in config.")
        return None

    print("Attempting LLM call to %s (%s) with prompt:" % (provider, model_name))
    print("--------------------------------------------------")
    print(prompt_text)
    print("--------------------------------------------------")

    try:
        if provider == "ollama_local":
            if not base_url:
                print("Error: base_url is required for ollama_local provider.")
                return None
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that generates Python code."},
                {"role": "user", "content": prompt_text}
            ]
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": llm_config.get('temperature', 0.7),
                "max_tokens": llm_config.get('max_tokens', 300),
                "stream": False
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer %s" % api_key if api_key else "Bearer ollama"
            }
            
            api_endpoint = base_url.rstrip('/') # Start with the base_url and remove any trailing slash

            # Check if it already ends with the full path
            if not api_endpoint.endswith("/v1/chat/completions"):
                # Check if it ends with just /v1
                if api_endpoint.endswith("/v1"):
                    api_endpoint += "/chat/completions"
                # Check if it's just the host:port (or http://host:port/)
                elif not "/v1" in api_endpoint:
                    api_endpoint += "/v1/chat/completions"

            response = await client.post(api_endpoint, json=payload, headers=headers, timeout=60.0)
            response.raise_for_status()
            json_response = response.json()
            
            if json_response.get("choices") and json_response["choices"][0].get("message"):
                return json_response["choices"][0]["message"].get("content", "").strip()
            else:
                print("Error: LLM response structure unexpected. Response: %s" % json_response)
                return "# Error: LLM response structure unexpected."
        # Add elif blocks here for other providers like 'openai', 'anthropic', 'google' if needed in the future
        # elif provider == "openai":
        #     if not api_key or api_key == "YOUR_API_KEY_HERE":
        #        print("Error: OpenAI API key is not configured.")
        #        return None
        #     client = OpenAI(api_key=api_key)
        #     completion_params = { ... similar to ollama ... }
        else:
            print("Unsupported LLM provider: %s" % provider)
            return "# Error: Unsupported LLM provider: %s" % provider

    except httpx.HTTPStatusError as e:
        print("HTTP error calling %s API: %s - Response: %s" % (provider, e, e.response.text))
        return "# Error during LLM call (HTTP): %s" % e
    except httpx.RequestError as e:
        print("Request error calling %s API: %s" % (provider, e))
        return "# Error during LLM call (Request): %s" % e
    except Exception as e:
        print("Generic error calling %s API: %s" % (provider, e))
        return "# Error during LLM call (Generic): %s" % e

async def main(): # Renamed and made async
    example_prompt = ("Given the Python function:\n" 
                      "```python\n"
                      "def solve():\n"
                      "    x = 1\n"
                      "    y = 1\n"
                      "    z = 1\n"
                      "    return (x * y) + z\n"
                      "```\n"
                      "Modify the values of x, y, and z (integers between -10 and 10) so that the function returns 42. "
                      "Only provide the complete modified Python code block for the solve() function."
                     )
    async with httpx.AsyncClient() as client: # Create client session
        llm_code = await get_llm_response(example_prompt, client)
    
    print("\nLLM response:")
    print(llm_code)

if __name__ == "__main__":
    asyncio.run(main()) # Run the async main 