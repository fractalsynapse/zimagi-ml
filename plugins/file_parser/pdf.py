from django.conf import settings
from pdf2image import pdfinfo_from_path, convert_from_path

from systems.plugins.index import BaseProvider

import pytesseract
import re


class Provider(BaseProvider('file_parser', 'pdf')):

  def parse_file(self, file_path):
    max_pages = pdfinfo_from_path(file_path)['Pages']
    batch_size = settings.PDF_OCR_BATCH_SIZE
    text = []

    for page in range(1, max_pages + 1, batch_size):
      doc = convert_from_path(file_path,
        dpi = settings.PDF_OCR_DPI,
        first_page = page,
        last_page = min(page + (batch_size - 1), max_pages)
      )
      for page_number, page_data in enumerate(doc):
        page_text = pytesseract.image_to_string(page_data).encode("utf-8").decode().strip()
        if page_text:
          text.append(re.sub(r'\n+\s+\n+', '\n\n', page_text))

    return "\n\n".join(text)
