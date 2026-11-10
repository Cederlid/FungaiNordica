import fitz
import ollama
from ollama import ResponseError


def extract_images_from_pdf(pdf_path: str, max_per_page: int = 2) -> list[dict]:
    doc = fitz.open(pdf_path)
    images = []
    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        count = 0
        for img_index, img in enumerate(image_list):
            if count >= max_per_page:
                break
            xref = img[0]
            base_image = doc.extract_image(xref)
            if base_image["width"] < 100 or base_image["height"] < 100:
                continue
            images.append({
                "page": page_num + 1,
                "index": img_index,
                "bytes": base_image["image"],
                "ext": base_image["ext"],
            })
            count += 1
    doc.close()
    return images


def caption_image(image_bytes: bytes, model: str = "llava") -> str | None:
    try:
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
    except ResponseError as e:
        print(f"    Warning: vision model error (skipping): {e}")
        return None


def process_spore_images(pdf_path: str, model: str = "llava") -> list[str]:
    images = extract_images_from_pdf(pdf_path)
    if not images:
        return []
    print(f"Found {len(images)} image(s), captioning with {model}...")
    captions = []
    for i, img in enumerate(images):
        print(f"  Image {i+1}/{len(images)} (page {img['page']})...")
        caption = caption_image(img["bytes"], model=model)
        if caption:
            text = f"[Spore illustration from page {img['page']}]: {caption}"
            captions.append(text)
    return captions
