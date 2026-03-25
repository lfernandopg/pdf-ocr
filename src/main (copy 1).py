import fitz  # PyMuPDF
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

class LegalPDFProcessor:
    def __init__(self):
        print("[INFO] Inicializando RapidOCR...")
        self.ocr = RapidOCR(common_config={"use_angle_cls": False})

    def _preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        black_ratio = np.sum(otsu == 0) / otsu.size
        
        if black_ratio > 0.15:
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            return cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 6
            )
        return gray

    def _get_structured_text(self, ocr_results, line_threshold=15):
        if not ocr_results:
            return ""

        # Ordenar de Arriba hacia Abajo (Y)
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
                    # Ordenar de Izquierda a Derecha (X)
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

    def process_pdf(self, pdf_path: str, dpi: int = 300, native_char_threshold: int = 100, is_inverted: bool = False) -> str:
        all_pages_content = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extracción nativa
                raw_native_text = page.get_text("text")
                if len(raw_native_text.replace(" ", "").replace("\n", "")) > native_char_threshold:
                    blocks = page.get_text("blocks")
                    blocks.sort(key=lambda b: b[1])
                    page_text = " ".join([" ".join(b[4].strip().split()) for b in blocks if b[4].strip()])
                else:
                    # Extracción por OCR
                    mat = fitz.Matrix(dpi / 72, dpi / 72).prerotate(page.rotation)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
                    img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    # MAGIA AQUÍ: Rotamos la imagen si sabemos que el escáner la tomó al revés
                    if is_inverted:
                        img_cv2 = cv2.rotate(img_cv2, cv2.ROTATE_180)
                    
                    processed_img = self._preprocess_image(img_cv2)
                    result, _ = self.ocr(processed_img)
                    
                    page_text = self._get_structured_text(result) if result else "[No se detectó texto]"
                
                all_pages_content.append(page_text)
                
            doc.close()
            return "\n\n".join(all_pages_content)

        except Exception as e:
            return f"[ERROR]: {str(e)}"

if __name__ == "__main__":
    FOLDER = "../docs/"
    INPUT_PDF = FOLDER + "TEST-EXHORTO.pdf"  
    OUTPUT_TXT = "resultado_legal_limpio.txt"
    
    processor = LegalPDFProcessor()
    
    # Activamos la bandera is_inverted=True para este documento específico
    texto_final = processor.process_pdf(INPUT_PDF, is_inverted=True)
    
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(texto_final)
    
    print(f"\n[ÉXITO] Proceso completado.")