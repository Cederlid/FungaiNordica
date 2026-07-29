import fitz
import ollama


def extract_images_from_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    images = []
    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            images.append({
                "page": page_num + 1,
                "index": img_index,
                "bytes": base_image["image"],
                "ext": base_image["ext"],
            })
    doc.close()
    return images


def caption_image(image_bytes: bytes, model: str = "llava") -> str:
    response = ollama.generate(
        model=model,
        prompt=(
            "You are analyzing a scientific mushroom illustration from Funga Nordica. "
            "Describe exactly what you see. Focus on:\n"
            "- Spore shapes and sizes\n"
            "- Any measurement labels or text (transcribe them exactly, e.g. '7-9 x 5-6 µm')\n"
            "- Species names or labels in the image\n"
            "- Any other visible annotations"
        ),
        images=[image_bytes],
    )
    return response["response"]


def process_spore_images(pdf_path: str, model: str = "llava") -> list[str]:
    images = extract_images_from_pdf(pdf_path)
    if not images:
        return []
    print(f"Found {len(images)} spore illustration(s), captioning with {model}...")
    captions = []
    for i, img in enumerate(images):
        print(f"  Captioning image {i+1}/{len(images)} (page {img['page']})...")
        caption = caption_image(img["bytes"], model=model)
        text = f"[Spore illustration from page {img['page']}]: {caption}"
        captions.append(text)
    return captions
