import fitz  # PyMuPDF
import cv2
import numpy as np
import logging
import re
import os
import difflib
from collections import Counter
from rapidocr_onnxruntime import RapidOCR

# Configuración de logging para entorno de producción
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LegalDocumentProcessor:
    def __init__(self):
        logger.info("Inicializando motor OCR...")
        self.ocr = RapidOCR(common_config={"use_angle_cls": True})

    def _preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # [NUEVO] Engrosamos ligeramente las letras para que el OCR no se salte textos finos o grises
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.erode(gray, kernel, iterations=1)
        
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        black_ratio = np.sum(otsu == 0) / otsu.size
        
        if black_ratio > 0.15:
            logger.debug(f"Imagen ruidosa (Densidad: {black_ratio:.1%}). Limpiando...")
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            return cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 6
            )
        
        logger.debug(f"Imagen limpia (Densidad: {black_ratio:.1%}). Procesando directo...")
        return gray

    def _get_structured_text(self, ocr_results, line_threshold=15):
        """Ordena las cajas de texto espacialmente y las unifica."""
        if not ocr_results:
            return ""

        ocr_results.sort(key=lambda x: sum([p[1] for p in x[0]]) / 4)

        lines = []
        current_line = []
        
        if ocr_results:
            last_y = sum([p[1] for p in ocr_results[0][0]]) / 4

            for res in ocr_results:
                curr_y = sum([p[1] for p in res[0]]) / 4

                if abs(curr_y - last_y) <= line_threshold:
                    current_line.append(res)
                else:
                    current_line.sort(key=lambda x: sum([p[0] for p in x[0]]) / 4)
                    lines.append(current_line)
                    current_line = [res]
                    last_y = curr_y
            
            current_line.sort(key=lambda x: sum([p[0] for p in x[0]]) / 4)
            lines.append(current_line)

        full_text = ""
        for line in lines:
            line_text = " ".join([item[1] for item in line])
            full_text += line_text + " "
            
        return full_text.strip()

    def process_pdf(self, pdf_path: str, dpi: int = 300, native_char_threshold: int = 100) -> str:
        """Procesa un PDF entero, decidiendo dinámicamente si usar extracción nativa u OCR."""
        all_pages_content = []
        
        try:
            doc = fitz.open(pdf_path)
            logger.info(f"Procesando documento: {pdf_path} ({len(doc)} páginas)")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                raw_native_text = page.get_text("text")
                if len(raw_native_text.replace(" ", "").replace("\n", "")) > native_char_threshold:
                    logger.info(f"Página {page_num + 1}: Texto digital detectado. Omitiendo OCR.")
                    blocks = page.get_text("blocks")
                    blocks.sort(key=lambda b: b[1])
                    page_text = " ".join([" ".join(b[4].strip().split()) for b in blocks if b[4].strip()])
                
                else:
                    logger.info(f"Página {page_num + 1}: Imagen detectada. Iniciando OCR...")
                    mat = fitz.Matrix(dpi / 72, dpi / 72).prerotate(page.rotation)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
                    img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    processed_img = self._preprocess_image(img_cv2)
                    result, _ = self.ocr(processed_img)
                    
                    page_text = self._get_structured_text(result) if result else "[No se detectó texto]"
                
                all_pages_content.append(page_text)
                
            doc.close()
            return "\n\n".join(all_pages_content)

        except Exception as e:
            logger.error(f"Fallo al procesar el PDF: {str(e)}")
            return ""

def calculate_accuracy(expected_text: str, generated_text: str):
    def normalize(text):
        return re.sub(r'[^\w\s]', '', text.lower())

    norm_expected = normalize(expected_text)
    norm_generated = normalize(generated_text)

    expected_words = norm_expected.split()
    generated_words = norm_generated.split()
    
    missing_words = []
    matches = 0
    
    # Unimos todo el texto generado para buscar sub-cadenas (Resuelve el problema de "Destinoexhorto")
    joined_generated = "".join(generated_words)
    
    for word in expected_words:
        # 1. Buscamos si la palabra está sola
        if word in generated_words:
            matches += 1
            generated_words.remove(word) # La consumimos para no contarla doble
        # 2. Buscamos si el OCR la pegó accidentalmente a otra palabra
        elif word in joined_generated:
            matches += 1
            joined_generated = joined_generated.replace(word, "", 1)
        else:
            missing_words.append(word)

    # Solo usamos el Acierto de Palabras, porque el Fuzzy Match no sirve si el orden cambia
    word_accuracy = (matches / len(expected_words)) * 100 if expected_words else 0

    return missing_words, word_accuracy

if __name__ == "__main__":
    FOLDER = "../docs/"
    INPUT_PDF = FOLDER + "TEST-EXHORTO.pdf"  
    OUTPUT_TXT = "resultado_legal_limpio.txt"
    EXPECTED_TXT = "expected_output.txt"
    
    # -- 1. Crear archivo de prueba si no existe --
    if not os.path.exists(EXPECTED_TXT):
        texto_prueba = (
            "Solicitud de Exhorto Resultado Exhorto Enviado Identificador exhorto: 3013947 "
            "Identificador solicitud exhorto: 20a64738-a6dd-4eea-b265-5a8fa974e2f2 "
            "N° Procedimiento: 0002538/2020 Destino exhorto: "
            "OFICINA DE REGISTRO Y REPARTO DE INSTRUCCIÓN DE MADRID "
            "Página 1 de 1 1 177 https://busprod.pnj.cgpj.es/pnj/ 01/10/2021"
        )
        with open(EXPECTED_TXT, "w", encoding="utf-8") as f:
            f.write(texto_prueba)
        logger.info(f"Archivo de prueba '{EXPECTED_TXT}' creado.")

    # -- 2. Procesar el PDF con mayor resolución (DPI=400) --
    processor = LegalDocumentProcessor()
    texto_final = processor.process_pdf(INPUT_PDF, dpi=400)
    
    # -- 3. Guardar el resultado --
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(texto_final)
    
    # -- 4. Calcular el porcentaje de acierto --
    # ¡IMPORTANTE! Leemos el contenido del archivo esperado
    with open(EXPECTED_TXT, "r", encoding="utf-8") as f:
        texto_esperado = f.read()
        
    # Pasamos los textos reales, no los nombres de los archivos
    faltantes, word_acc = calculate_accuracy(texto_esperado, texto_final)
    
    print("\n" + "="*60)
    print("[ÉXITO] Proceso completado.")
    print("-" * 60)
    print(f"-> Acierto Real para el LLM (Búsqueda de subcadenas):  {word_acc:.2f}%")
    
    if faltantes:
        print("\n[ALERTA] El OCR omitió por completo estas palabras:")
        conteo_faltantes = Counter(faltantes)
        for palabra, cantidad in conteo_faltantes.items():
            print(f"   - '{palabra}' (Faltó {cantidad} vez/veces)")
    else:
        print("\n[PERFECTO] Se detectaron todas las palabras clave necesarias.")
    print("="*60 + "\n")