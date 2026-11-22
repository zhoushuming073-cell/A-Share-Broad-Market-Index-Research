"""Build the integrated report from its editable manuscript and repository figures."""
import argparse
import importlib.util
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = HERE / "research_report.md"
BASE = REPO / "experiments/four_pair_tests_v2/build_word.py"
ARCHIVE = "https://github.com/zhoushuming073-cell/A-Share-Broad-Market-Index-Research/blob/6a4c8f4e65b968814e34e0417cbbdbaaf3d35cc1/"


def prepare(text, folder):
    """Render SVGs at print resolution and resolve repository links for readers."""
    from playwright.sync_api import sync_playwright

    def resolve(match):
        image, label, target = match.groups()
        if target.startswith(("https://", "http://", "#")):
            return match[0]
        src = (HERE / target).resolve()
        if not src.is_file():
            raise FileNotFoundError(src)
        if not image:
            return f"[{label}]({ARCHIVE}{quote(src.relative_to(REPO).as_posix())})"
        filename = f"chart_{len(images):02d}.png"
        dest = folder / filename
        if src.suffix == ".svg":
            xml = ET.parse(src).getroot()
            _, _, w, h = map(float, xml.attrib["viewBox"].split())
            width = 1920
            height = round(width * h / w)
            page.set_viewport_size({"width": width, "height": height})
            page.goto(src.as_uri())
            page.evaluate("""() => {
                const svg = document.documentElement;
                svg.setAttribute('width','100%'); svg.setAttribute('height','100%');
                svg.style.background = 'white';
            }""")
            page.evaluate("document.fonts.ready")
            page.screenshot(path=str(dest), animations="disabled")
        else:
            shutil.copyfile(src, dest)
        images.append(filename)
        return f"![]({filename})"

    images = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge")
        try:
            page = browser.new_page()
            return re.sub(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)", resolve, text)
        finally:
            browser.close()


def build(folder):
    spec = importlib.util.spec_from_file_location("paired_word", BASE)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    base.ROOT = folder
    manuscript = prepare(SOURCE.read_text(encoding="utf-8"), folder)
    (folder / "research_report.md").write_text(manuscript, encoding="utf-8")
    path = base.build()
    doc = Document(path)
    doc.core_properties.title = "A股宽基指数的投资价值与多空收益稳定性"
    doc.core_properties.subject = "指数配置、相对价值、关键交易日与历史选择偏差"
    doc.core_properties.author = "A股宽基指数研究项目"
    header = doc.sections[0].header.paragraphs[0]
    header.clear()
    header.add_run("A股宽基指数的投资价值与多空收益稳定性").font.size = Pt(9)
    h2 = doc.styles["Heading 2"]
    base.font(h2, 13, "黑体")
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(7)
    doc.styles["Title"].font.color.rgb = RGBColor.from_string("132731")
    for p in doc.paragraphs:
        if p.style.name == "Heading 1":
            p.paragraph_format.page_break_before = p.text != "讨论与结论"
        elif p.style.name == "Caption":
            p.paragraph_format.keep_together = True
        elif p.style.name in ("Title", "Subtitle", "Date"):
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for table in doc.tables:
        for row_num, row in enumerate(table.rows):
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = row_num == 0
    for shape in doc.inline_shapes:
        factor = min(Cm(16.0) / shape.width, Cm(14.0) / shape.height)
        shape.width = round(shape.width * factor)
        shape.height = round(shape.height * factor)
    doc.save(path)
    return path


def finalize(path, pdf):
    import win32com.client
    word = win32com.client.DispatchEx("Word.Application")
    try:
        word.Visible = False
        word.DisplayAlerts = 0
        report = word.Documents.Open(str(path), AddToRecentFiles=False)
        report.Fields.Update()
        report.Repaginate()
        report.TablesOfContents(1).Update()
        report.Repaginate()
        report.TablesOfContents(1).UpdatePageNumbers()
        report.Save()
        if pdf:
            report.ExportAsFixedFormat(str(pdf.resolve()), 17)
        report.Close(False)
    finally:
        word.Quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, help="Optional pagination proof")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        compiled = build(Path(temp))
        finalize(compiled, args.pdf)
        shutil.copyfile(compiled, REPO / "research_report.docx")

