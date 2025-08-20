import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def save_all_texts(input_dir="data", output_dir="data"):
    for i in range(1, 6):  
        pdf_file = f"{i}.pdf" 
        pdf_path = os.path.join(input_dir, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        with open(os.path.join(output_dir, f"{i}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ Extracted: {i}.txt")

if __name__ == "__main__":
    save_all_texts()
