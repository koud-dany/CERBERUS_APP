try:
    import openai
    print("✅ OpenAI library is available")
    print("Version:", openai.__version__)
except ImportError as e:
    print("❌ OpenAI library not installed:", e)
    print("Install with: pip install openai")
