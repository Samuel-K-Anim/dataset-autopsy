from fpdf import FPDF
import pandas as pd

class PDFReport(FPDF):
    #--- PDF Report Class ---
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Dataset Autopsy & Impact Report', 0, 1, 'C')
        self.ln(5)

    #--- Footer ---
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    #--- Text Cleaning ---
    def clean_text(self, text):
        """Removes characters that aren't supported by Latin-1 (like emojis)"""
        if isinstance(text, str):
            return text.encode('latin-1', 'ignore').decode('latin-1')
        return str(text)

    #--- Chapter Title ---
    def chapter_title(self, title):
        clean_title = self.clean_text(title)
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, clean_title, 0, 1, 'L', 1)
        self.ln(4)

    #--- Chapter Body ---
    def chapter_body(self, body):
        clean_body = self.clean_text(body)
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 8, clean_body)
        self.ln()

    #--- Add Table Data ---
    def add_table_data(self, df, title):
        self.set_font('Arial', 'B', 10)
        self.cell(0, 10, self.clean_text(title), 0, 1)
        self.set_font('Arial', '', 9)
        table_txt = df.to_string()
        clean_table = self.clean_text(table_txt)
        self.multi_cell(0, 5, clean_table)
        self.ln()


def generate_pdf_report(filename, stats, missing_df, outliers_df, skew_df, recommendations, score, change_log=None, impact_summary=None):
    pdf = PDFReport()
    pdf.add_page()
    
    # 1. SUMMARY
    pdf.chapter_title(f"1. Executive Summary: {filename}")
    summary_text = (
        f"ML Readiness Score: {score}/100\n"
        f"Rows: {stats['rows']} | Columns: {stats['cols']}\n"
        f"Missing Cells: {stats['missing_cells']}"
    )
    pdf.chapter_body(summary_text)
    
    # 2. CHANGE LOG & IMPACT ANALYSIS
    if change_log or impact_summary:
        pdf.chapter_title("2. Treatment & Impact Analysis")
        if change_log:
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 10, "Actions Taken:", 0, 1)
            pdf.set_font('Arial', '', 10)
            for action in change_log:
                pdf.cell(0, 8, f"- {pdf.clean_text(action)}", 0, 1)
            pdf.ln(2)
            
        if impact_summary:
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 10, "Impact on Data Health:", 0, 1)
            pdf.set_font('Arial', '', 10)
            for impact in impact_summary:
                pdf.cell(0, 8, f"- {pdf.clean_text(impact)}", 0, 1)
            pdf.ln()

    # 3. ISSUES
    pdf.chapter_title("3. Detected Issues")
    if not missing_df.empty:
        pdf.add_table_data(missing_df.head(5), "Missing Data:")
    else:
        pdf.chapter_body("No missing values found.")
        
    # 4. RECOMMENDATIONS
    pdf.chapter_title("4. Recommendations")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            clean_rec = rec.replace("**", "")
            pdf.cell(0, 8, f"{i}. {pdf.clean_text(clean_rec)}", 0, 1)
    return pdf.output(dest='S').encode('latin-1', 'ignore')