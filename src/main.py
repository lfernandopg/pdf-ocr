import fitz  # PyMuPDF
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

class PDFRapidOCR:
    def __init__(self):
        print("[INFO] Inicializando RapidOCR con Clasificador de Ángulo...")
        # Activamos use_angle_cls=True para corregir hojas volteadas o rotadas
        self.ocr = RapidOCR(common_config={"use_angle_cls": True})

    def _sort_boxes(self, dt_boxes):
        """
        Ordena las cajas de texto de arriba hacia abajo y de izquierda a derecha.
        """
        # Obtenemos el índice de ordenación basado en la coordenada Y superior
        indices = sorted(range(len(dt_boxes)), key=lambda i: dt_boxes[i][0][1])
        return indices

    def extract_text_from_pdf(self, pdf_path: str, dpi: int = 300) -> str:
        full_text = []
        
        try:
            print(f"[INFO] Abriendo documento: {pdf_path}")
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                
                for page_num in range(total_pages):
                    print(f"[INFO] Procesando página {page_num + 1} de {total_pages}...")
                    page = doc[page_num]
                    
                    # 1. Renderizar la página
                    pix = page.get_pixmap(dpi=dpi, alpha=False)
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
                    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # 2. Pasar la imagen por RapidOCR
                    # use_angle_cls corregirá la rotación de 180° automáticamente
                    result, _ = self.ocr(img_cv2)
                    
                    page_text = f"--- INICIO PÁGINA {page_num + 1} ---\n"
                    
                    if result:
                        # 3. Ordenar resultados para que el flujo sea humano (arriba -> abajo)
                        # RapidOCR devuelve: [ [box], text, score ]
                        # Ordenamos por la coordenada Y del primer punto de la caja (box[0][1])
                        result.sort(key=lambda x: x[0][0][1])
                        
                        for line in result:
                            # line[0] = caja, line[1] = texto, line[2] = confianza
                            text = line[1]
                            page_text += f"{text}\n"
                            
                    page_text += f"--- FIN PÁGINA {page_num + 1} ---\n"
                    full_text.append(page_text)
                    
            return "\n".join(full_text)

        except Exception as e:
            print(f"[ERROR] Ocurrió un error al procesar el PDF: {e}")
            return ""

if __name__ == "__main__":
    PDF_FILE = "TEST-EXHORTO.pdf"
    
    procesador = PDFRapidOCR()
    
    print("\n[INFO] Iniciando extracción con corrección de rotación...")
    texto_extraido = procesador.extract_text_from_pdf(PDF_FILE)
    
    print("\n================ RESULTADO DEL OCR ================\n")
    print(texto_extraido)
    
    with open("resultado_ocr.txt", "w", encoding="utf-8") as f:
        f.write(texto_extraido)
    print("\n[INFO] El resultado se ha guardado en 'resultado_ocr.txt'")