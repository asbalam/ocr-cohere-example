# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cohere
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from io import BytesIO
import os
import json
import re

# 🚀 Inicializar Flask
app = Flask(__name__)
CORS(app)

# 🔐 Cargar clave Cohere desde variable de entorno
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("⚠️ Falta la variable de entorno COHERE_API_KEY")

co = cohere.Client(COHERE_API_KEY)

# 📌 Endpoint principal
@app.route('/ocr-curp', methods=['POST'])
def ocr_curp():
    if 'file' not in request.files:
        return jsonify({"error": "No se envió un archivo"}), 400

    file = request.files['file']
    content_type = file.content_type

    try:
        # 📄 Convertir PDF o abrir imagen
        if content_type == "application/pdf":
            pages = convert_from_bytes(file.read())
            image = pages[0]
        else:
            image = Image.open(file)
    except Exception as e:
        return jsonify({"error": f"Error procesando archivo: {str(e)}"}), 500

    # 🧾 Extraer texto con OCR
    try:
        text = pytesseract.image_to_string(image, lang="spa")
    except Exception as e:
        return jsonify({"error": f"Error en OCR: {str(e)}"}), 500

    # 🧠 Prompt para Cohere
    prompt = f"""
    Este es un texto obtenido de un documento con datos personales.
    Extrae la información y responde SOLO con un JSON plano, sin explicaciones.
    Formato esperado:
    {{
      "nombre": "",
      "apellido_paterno": "",
      "apellido_materno": "",
      "fecha": "DD/MM/YYYY",
      "calle_y_numero": "",
      "colonia": "",
      "ciudad_municipio": "",
      "estado": "",
      "codigo_postal": "",
      "pais_nacimiento": "",
      "nacionalidad": "",
      "fecha_nacimiento": "DD/MM/YYYY",
      "rfc": "",
      "correo_electronico": "",
      "telefono": "",
      "ocupacion": "",
      "origen_recursos": "",
      "ha_desempenado_cargo_en_gobierno": "Sí" o "No"
    }}

    Texto a analizar:
    {text}
    """

    try:
        response = co.chat(
            model="command-r-plus",
            message=prompt
        )

        raw = response.text.strip()
        cleaned = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()

        # Intentar parsear el JSON
        parsed = json.loads(cleaned)
        return jsonify(parsed)

    except json.JSONDecodeError:
        return jsonify({"error": "JSON inválido devuelto por el modelo", "raw": raw}), 500
    except Exception as e:
        return jsonify({"error": f"Error en Cohere: {str(e)}"}), 500

# 🖥️ Ejecutar servidor local
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
