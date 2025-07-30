import os
import json

import pydantic
import polars as pl
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from .env file for secure API key management
load_dotenv()

# --- CHOOSE YOUR PROVIDER ---
# Set to "openai" or "gemini" to select the API provider.
PROVIDER = "gemini"  # <-- CHANGE THIS TO "openai" OR "gemini"

if PROVIDER == "gemini":
    # For Gemini, you need an API key from Google AI Studio (https://aistudio.google.com/app/apikey)
    # and the correct base URL. Add these to your .env file:
    # GEMINI_API_KEY="your_google_ai_studio_key"
    # GEMINI_API_BASE_URL="https://generativelanguage.googleapis.com/v1beta"
    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    if not api_key:
        raise ValueError("For Gemini provider, GEMINI_API_KEY must be set in your .env file.")
    
    print("Using Gemini provider.")
    client = OpenAI(api_key=api_key, base_url=base_url)
    # The model name for Gemini API via an OpenAI-compatible endpoint is often the full path.
    MODEL = "models/gemini-2.0-flash"

elif PROVIDER == "openai":
    # For OpenAI, the client uses the OPENAI_API_KEY from your .env file by default.
    print("Using OpenAI provider.")
    client = OpenAI()
    MODEL = "gpt-4o"  # Or "gpt-4-turbo"
else:
    raise ValueError(f"Unsupported provider: '{PROVIDER}'. Choose 'openai' or 'gemini'.")
    
# --- Data Structure Definition using Pydantic ---
# We define the exact structure we want to extract from the PDF.
# Pydantic models ensure the data returned by the AI is in the correct format.
# Using `Field(alias=...)` lets us use user-friendly names in the output
# while having clean, Python-friendly variable names in our code.

class PowerCabinetSpecs(pydantic.BaseModel):
    """Extracted specifications for a power cabinet or all-in-one EV charging unit."""
    model: str | None = pydantic.Field(
        None, description="The model name or number of the unit."
    )
    ingress_protection: str | None = pydantic.Field(
        None, serialization_alias="Ingress Protection", description="The IP rating (e.g., IP55)."
    )
    nema_enclosure: str | None = pydantic.Field(
        None, serialization_alias="NEMA Enclosure", description="The NEMA enclosure type (e.g., NEMA 3R)."
    )
    operating_temperature_range: str | None = pydantic.Field(
        None, serialization_alias="Operating temperature range", description="The valid operating temperature, e.g., -30°C to +50°C."
    )
    operating_humidity: str | None = pydantic.Field(
        None, serialization_alias="Operating Humidity", description="The valid operating humidity range, e.g., 5% to 95%."
    )
    operating_altitude: str | None = pydantic.Field(
        None, serialization_alias="Operating Altitude", description="Maximum operating altitude, e.g., < 2000m."
    )
    noise_level: str | None = pydantic.Field(
        None, serialization_alias="Noise Level", description="The noise level in decibels, e.g., < 65 dB."
    )
    dimensions: str | None = pydantic.Field(
        None, serialization_alias="Dimensions (H × W × D)", description="The physical dimensions of the unit, e.g., '1800 × 600 × 800 mm'."
    )
    enclosure_protection: str | None = pydantic.Field(
        None, serialization_alias="Enclosure Protection", description="General enclosure protection rating, may overlap with IP/NEMA."
    )
    display: str | None = pydantic.Field(
        None, description="Details about the display, e.g., '7-inch LCD touchscreen'."
    )
    compliance: str | None = pydantic.Field(
        None, description="List of standards the unit complies with."
    )
    certification: str | None = pydantic.Field(
        None, description="List of certifications the unit has obtained (e.g., CE, UL)."
    )
    emc_emi: str | None = pydantic.Field(
        None, serialization_alias="EMC/EMI", description="EMC/EMI compliance standards."
    )
    payment_terminal: str | None = pydantic.Field(
        None, serialization_alias="Payment terminal", description="Details about the payment system (e.g., RFID, Credit Card)."
    )
    dc_output_v: str | None = pydantic.Field(
        None, serialization_alias="DC-output V", description="The DC output voltage range, e.g., '200-1000 VDC'."
    )
    dc_output_a: str | None = pydantic.Field(
        None, serialization_alias="DC-output A", description="The DC output current, e.g., 'Max 200 A'."
    )
    liquid_cooling: bool | None = pydantic.Field(
        None, description="Whether the unit uses liquid cooling."
    )
    current: str | None = pydantic.Field(
        None, description="Input or output current specifications, if different from DC-output A."
    )
    cms: str | None = pydantic.Field(
        None, serialization_alias="CMS", description="Charging Management System details (e.g., OCPP 1.6J)."
    )
    cable_length: str | None = pydantic.Field(
        # Note: Corrected typo from "lenght" to "length"
        None, serialization_alias="Cable length", description="The length of the charging cable(s)."
    )
    cable_reach: str | None = pydantic.Field(
        None, serialization_alias="Cable reach", description="The effective reach of the cable."
    )
    dispenser_link: str | None = pydantic.Field(
        None, serialization_alias="Dispenser Link", description="Information on how the cabinet links to dispensers."
    )

    class Config:
        # This allows creating the model using aliases
        populate_by_name = True

def convert_schema_for_gemini(properties: dict) -> dict:
    fixed = {}
    for k, v in properties.items():
        if "anyOf" in v:
            # Extract the type that is not null
            for option in v["anyOf"]:
                if option.get("type") != "null":
                    v["type"] = option["type"]
                    break
            v.pop("anyOf", None)
        fixed[k] = v
    return fixed

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from all pages of a PDF file.

    Args:
        pdf_path: The file path to the PDF.

    Returns:
        A single string containing all the text from the PDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file was not found at: {pdf_path}")
    
    print(f"Reading text from '{pdf_path}'...")
    reader = PdfReader(pdf_path)
    all_text = []
    for i, page in enumerate(reader.pages):
        try:
            all_text.append(page.extract_text() or "")
        except Exception as e:
            print(f"Could not extract text from page {i+1}: {e}")
    
    print("Finished reading PDF text.")
    return "\n".join(all_text)


def get_specs_from_text(text: str) -> PowerCabinetSpecs:
    """
    Uses OpenAI's function calling to extract structured data from text.

    Args:
        text: The text content from the PDF.

    Returns:
        A Pydantic object containing the extracted specifications.
    """
    print("Sending text to OpenAI for extraction...")
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at extracting technical specifications from documents for EV charging power cabinets and all-in-one units. Please extract the information based on the user's request from the provided text. If a value is not found, leave it as null.",
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "extract_power_cabinet_specs",
                        "description": "Extracts technical specifications from a document.",
                        "parameters": {
                                        "type": "object",
                                        "properties": convert_schema_for_gemini(PowerCabinetSpecs.model_json_schema(by_alias=True)["properties"]),
                                        "required": []  # Optional: you can explicitly set required fields
                                    },
                    },
                }
            ],
            tool_choice={
                "type": "function",
                "function": {"name": "extract_power_cabinet_specs"},
            },
        )

        tool_call = response.choices[0].message.tool_calls[0]
        if tool_call.function.name == "extract_power_cabinet_specs":
            print("OpenAI returned structured data. Parsing results...")
            arguments = json.loads(tool_call.function.arguments)
            return PowerCabinetSpecs.model_validate(arguments)
        else:
            raise ValueError(f"Expected function 'extract_power_cabinet_specs' but got '{tool_call.function.name}'")

    except Exception as e:

        import pprint
        print("\n--- Final Schema Sent to Gemini ---")
        pprint.pprint(PowerCabinetSpecs.model_json_schema(by_alias=True)["properties"])
        print("------------------------------------\n")
        print(f"An error occurred during OpenAI API call: {e}")
        raise


def main():
    """Main function to run the PDF spec extraction process."""
    # --- IMPORTANT ---
    # Replace this with the actual path to your PDF file.
    pdf_file_path = r"C:\Users\Nebil Ibrahim\Downloads\Kempower-L3-charger-specs.pdf"

    try:
        pdf_text = extract_text_from_pdf(pdf_file_path)

        if not pdf_text.strip():
            print("Could not extract any text from the PDF. It might be an image-based PDF.")
            print("Consider using an OCR tool first to convert the PDF to text.")
            return

        extracted_specs = get_specs_from_text(pdf_text)
        

        print("\n--- Extracted Specifications ---")
        print(json.dumps(extracted_specs.model_dump(by_alias=True, exclude_none=True), indent=2))
        print("------------------------------")
        
        # Convert to Polars DataFrame and save to CSV
        specs_dict = extracted_specs.model_dump(by_alias=True, exclude_none=False)
        df = pl.DataFrame([specs_dict])
        output_csv_path = "extracted_specs.csv"
        df.write_csv(output_csv_path)
        print(f"\nExtracted specifications saved to '{output_csv_path}'")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please update the 'pdf_file_path' variable in the `main` function to point to your PDF.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()