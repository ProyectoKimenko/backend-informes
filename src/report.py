from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import tempfile

class Report:
    def __init__(self, title, place_name="", template_dir='templates'):
        self.title = title
        self.place_name = place_name
        self.sections = []
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.created_at = datetime.now()

    def add_section(self, section):
        self.sections.append(section)

    def render(self):
        main_template = self.env.get_template('pdf_main.html')
        rendered_html = main_template.render(
            title=self.title,
            created_at=self.created_at,
            place_name=self.place_name,
            sections=self.sections
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            pdf_file = temp_file.name
            HTML(string=rendered_html).write_pdf(pdf_file)
        return pdf_file