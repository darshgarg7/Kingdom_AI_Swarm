from input_layer import InputLayer
from concurrent.futures import ThreadPoolExecutor

def preprocess_and_validate(input_layer, raw_input):
    """
    Preprocesses and validates raw input.
    Args:
        input_layer (InputLayer): Instance of the InputLayer class.
        raw_input (str): Raw input string to preprocess and validate.
    Returns:
        List[str]: Preprocessed tokens if valid, otherwise None.
    """
    try:
        tokens = input_layer.preprocess(raw_input)
        print(f"Preprocessed Tokens: {tokens}")
        
        is_valid = input_layer.validate_input(tokens)
        if not is_valid:
            raise ValueError("Input validation failed.")
        
        return tokens
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

def preprocess_inputs_concurrently(input_layer, raw_inputs):
    """
    Preprocesses multiple inputs concurrently using a thread pool.
    Args:
        input_layer (InputLayer): Instance of the InputLayer class.
        raw_inputs (List[str]): List of raw input strings to preprocess.
    Returns:
        List[List[str]]: List of preprocessed tokens for each input.
    """
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda raw: preprocess_and_validate(input_layer, raw), raw_inputs))
    return results

def fetch_external_data(input_layer, api_url):
    """
    Fetches external data using the IntegrationLayer.
    Args:
        input_layer (InputLayer): Instance of the InputLayer class.
        api_url (str): URL of the external API to fetch data from.
    Returns:
        Dict: Fetched external data, or an empty dictionary if an error occurs.
    """
    try:
        external_data = input_layer.fetch_external_data(api_url)
        if external_data:
            print(f"Fetched External Data: {external_data}")
            return external_data
        else:
            print("No external data fetched.")
            return {}
    except Exception as e:
        print(f"Error fetching external data: {e}")
        return {}

def generate_scenario(input_layer, prompt):
    """
    Generates a realistic scenario using GPT-4 or another LLM.
    Args:
        input_layer (InputLayer): Instance of the InputLayer class.
        prompt (str): Description of the scenario to generate.
    Returns:
        str: Generated scenario details, or an empty string if an error occurs.
    """
    try:
        scenario = input_layer.generate_scenario(prompt)
        print(f"Generated Scenario: {scenario}")
        return scenario
    except Exception as e:
        print(f"Error generating scenario: {e}")
        return ""

def main():
    print("Starting Preprocessing Module...")

    input_layer = InputLayer()

    raw_inputs = [
        "Optimize traffic in smart cities!",
        "Reduce congestion in urban areas."
    ]
    print(f"Raw Inputs: {raw_inputs}")

    tokens_list = preprocess_inputs_concurrently(input_layer, raw_inputs)
    print(f"Processed Tokens List: {tokens_list}")

    api_url = "https://api.example.com/traffic"
    external_data = fetch_external_data(input_layer, api_url)

    if external_data:
        print(f"Using external data: {external_data}")
        # Example: Pass external_data to generate scenarios dynamically
        for raw_input in raw_inputs:
            enhanced_prompt = f"{raw_input} ({external_data.get('region', 'unknown region')})"
            generate_scenario(input_layer, enhanced_prompt)

if __name__ == "__main__":
    main()
