import os
import json
import time
import argparse
import base64
import fitz  # PyMuPDF

import pydantic
import polars as pl
from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from .env file for secure API key management
load_dotenv()

# --- CHOOSE YOUR PROVIDER ---
# Set to "gemini" for vision capabilities.
PROVIDER = "gemini"

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
    # Use a model that supports vision input
    MODEL = "models/gemini-2.5-flash" 
else:
    raise ValueError(f"Unsupported provider: '{PROVIDER}'. This script requires 'gemini' for vision capabilities.")
    
# --- Data Structure Definition using Pydantic ---
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
        None, serialization_alias="Cable length", description="The length of the charging cable(s)."
    )
    cable_reach: str | None = pydantic.Field(
        None, serialization_alias="Cable reach", description="The effective reach of the cable."
    )
    dispenser_link: str | None = pydantic.Field(
        None, serialization_alias="Dispenser Link", description="Information on how the cabinet links to dispensers."
    )

    class Config:
        populate_by_name = True

def convert_schema_for_gemini(properties: dict) -> dict:
    fixed = {}
    for k, v in properties.items():
        if "anyOf" in v:
            for option in v["anyOf"]:
                if option.get("type") != "null":
                    v["type"] = option["type"]
                    break
            v.pop("anyOf", None)
        fixed[k] = v
    return fixed

def convert_pdf_pages_to_images(pdf_path: str) -> list[str]:
    """
    Converts each page of a PDF into a base64 encoded image string.

    Args:
        pdf_path: The file path to the PDF.

    Returns:
        A list of base64 encoded image strings.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file was not found at: {pdf_path}")

    print(f"Converting PDF pages to images from '{pdf_path}'...")
    doc = fitz.open(pdf_path)
    base64_images = []

    for page_index in range(len(doc)):
        page: fitz.Page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=200) # Higher DPI for better quality
        img_bytes = pix.tobytes("jpeg")
        base64_images.append(base64.b64encode(img_bytes).decode('utf-8'))

    doc.close()
    print(f"Successfully converted {len(base64_images)} pages to images.")
    return base64_images

def get_specs_from_images(images: list[str]) -> PowerCabinetSpecs:
    """
    Uses a multimodal AI to extract structured data from a list of images.

    Args:
        images: A list of base64 encoded image strings.

    Returns:
        A Pydantic object containing the extracted specifications.
    """
    print(f"Sending {len(images)} images to Gemini for extraction...")
    
    # Construct the messages for the API call
    messages = [
        {
            "role": "system",
            "content": "You are an expert at extracting technical specifications from documents for EV charging power cabinets and all-in-one units. Please extract the information based on the user's request from the provided text. If a value is not found, leave it as null.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract the specifications from the following page images:"
                },
            ] + [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                } for img in images
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "extract_power_cabinet_specs",
                        "description": "Extracts technical specifications from a document.",
                        "parameters": {
                            "type": "object",
                            "properties": convert_schema_for_gemini(PowerCabinetSpecs.model_json_schema(by_alias=True)["properties"]),
                            "required": []
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
            print("Gemini returned structured data. Parsing results...")
            arguments = json.loads(tool_call.function.arguments)
            return PowerCabinetSpecs.model_validate(arguments)
        else:
            raise ValueError(f"Expected function 'extract_power_cabinet_specs' but got '{tool_call.function.name}'")

    except Exception as e:
        print(f"An error occurred during Gemini API call: {e}")
        raise

def main():
    """Main function to run the PDF image-based spec extraction process."""
    parser = argparse.ArgumentParser(
        description="Extract technical specifications from a PDF data sheet using vision capabilities."
    )
    parser.add_argument(
        "pdf_file",
        nargs="?",
        default=r"C:\Users\Nebil Ibrahim\Downloads\ABB_Emobility_Terra-360_UL_Data-Sheet_A.pdf",
        help="Path to the input PDF file. Defaults to a sample Terra 360 datasheet."
    )
    parser.add_argument(
        "--output", "-o",
        default="extracted_specs_vision.csv",
        help="Path for the output CSV file. Defaults to 'extracted_specs_vision.csv'."
    )
    args = parser.parse_args()

    try:
        pdf_images = convert_pdf_pages_to_images(args.pdf_file)

        if not pdf_images:
            print("Could not convert any pages from the PDF to images.")
            return
        start = time.time()
        extracted_specs = get_specs_from_images(pdf_images)
        print("Elapsed Time:", time.time() - start, " seconds")
        print("\n--- Extracted Specifications ---")
        print(json.dumps(extracted_specs.model_dump(by_alias=True, exclude_none=True), indent=2))
        print("------------------------------")

        # Convert to Polars DataFrame and save to CSV
        specs_dict = extracted_specs.model_dump(by_alias=True, exclude_none=False)
        df = pl.DataFrame([specs_dict])
        df.write_csv(args.output)
        print(f"\nExtracted specifications saved to '{args.output}'")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please check if the provided PDF file path is correct.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()