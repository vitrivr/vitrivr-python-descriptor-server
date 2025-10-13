import base64
from io import BytesIO
from apiflask import APIBlueprint
from flask import request, jsonify
from PIL import Image
import pytesseract

ocr = APIBlueprint('ocr', __name__, tag='OCR')


@ocr.post('/extract/ocr')
@ocr.doc(
    summary="OCR endpoint that extracts text from an image encoded as a base64 data URL."
)
def extract_text():
    """
    Extract text from an image sent as a base64-encoded data URL in the 'data' form field.
    """
    data = request.form.get('data', '')

    if not data or "base64," not in data:
        return jsonify({'error': 'Missing or invalid base64 image data'}), 400

    try:
        # Decode base64 data (strip the header like data:image/png;base64,)
        header, encoded = data.split("base64,", 1)
        image = Image.open(BytesIO(base64.b64decode(encoded)))

        # Perform OCR using Tesseract
        extracted_text = pytesseract.image_to_string(image)

        return jsonify({
            'text': extracted_text.strip()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
