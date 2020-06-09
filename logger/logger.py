class Logger:
    def __init__(self):
        self.active_section = LoggerSection('head')
        self.logs_head = self.active_section
        self.parent_section = None

    def add_section(self, section_name):
        sub_section = LoggerSection(section_name)
        self.active_section.add_subsection(sub_section)
        self.parent_section = self.active_section
        self.active_section = sub_section

    def add_log(self, log):
        self.active_section.add_content(log)

    def to_string(self):
        pass


class LoggerSection:
    def __init__(self, section_name):
        self.section_name = section_name
        self.sub_sections = []
        self.section_content = []

    def add_subsection(self, subsection):
        self.sub_sections.append(subsection)

    def add_content(self, message):
        self.section_content.append(message)
