# llm_context_os/tools/builtin_weather.py
def get_weather(location: str, unit: str = "celsius") -> str:
    """Placeholder function to get weather.
    In a real scenario, this would call a weather API.
    """
    print(f"[Tool] get_weather called for {location} in {unit}")
    if "new york" in location.lower():
        return f"The weather in New York is 22 degrees {unit}."
    elif "london" in location.lower():
        return f"The weather in London is 15 degrees {unit}."
    else:
        return f"Sorry, I don't have weather information for {location}."
