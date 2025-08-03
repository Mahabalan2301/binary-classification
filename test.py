from tensorflow import keras

try:
    # Attempt to load the model
    model = keras.models.load_model("cat_vs_dogs_model.keras")
    
    # Print a summary to confirm it's loaded correctly
    model.summary()
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:")
    print(e)
