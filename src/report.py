from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import tempfile
import base64
from PIL import Image
from io import BytesIO
import os

class Report:
    def __init__(self, title, place_name="", template_dir='templates'):
        self.title = title
        self.place_name = place_name
        self.sections = []
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.created_at = datetime.now()
        
        # Add logo path
        self.logo_path = os.path.abspath('static/logo-kimenko.png')

        
        
        # Optimize background image before encoding
        with Image.open('static/mountain.jpg') as img:
            # Resize image if too large (e.g., to max 1500px width)
            max_width = 500
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Save optimized image to bytes
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=60, optimize=True)
            self.background_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    def add_section(self, section):
        self.sections.append(section)

    def render(self):
        main_template = self.env.get_template('pdf_main.html')
        rendered_html = main_template.render(
            title=self.title,
            created_at=self.created_at,
            place_name=self.place_name,
            sections=self.sections,
            background_image_base64=self.background_image_base64,
            logo_path=self.logo_path
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            pdf_file = temp_file.name
            HTML(string=rendered_html).write_pdf(
                pdf_file,
                optimize_images=True,
                jpeg_quality=50,
                compress=True
            )
        return pdf_file