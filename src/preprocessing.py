from src.input_layer import InputLayer

def main():
    print("Starting Preprocessing Module...")
    
    # Initialize Input Layer
    input_layer = InputLayer()
    
    # Example raw input (can be replaced with real-time data or API calls)
    raw_input = "Optimize traffic in smart cities!"
    print(f"Raw Input: {raw_input}")
    
    # Preprocess the input
    tokens = input_layer.preprocess(raw_input)
    print(f"Preprocessed Tokens: {tokens}")
    
    # Validate the input
    is_valid = input_layer.validate_input(tokens)
    if not is_valid:
        print("Preprocessing failed due to invalid input.")
        return
    
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()
    