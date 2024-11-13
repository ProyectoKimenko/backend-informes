class ReportSection:
    def __init__(self, title, template_name):
        self.title = title
        self.template_name = template_name
        self.data = {}

    def add_data(self, key, value):
        self.data[key] = value

class WeekdaySection(ReportSection):
    def __init__(self, title):
        super().__init__(title, "weekday.html")

class WeekendSection(ReportSection):
    def __init__(self, title):
        super().__init__(title, "weekday.html")

class ComparisonSection(ReportSection):
    def __init__(self, title):
        super().__init__(title, "comparison.html")