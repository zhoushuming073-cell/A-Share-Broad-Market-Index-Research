"""Compile the editable Markdown to Word; use desktop Word to paginate and update fields."""
from pathlib import Path
import tempfile
import argparse
import pypandoc
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent

def elem(tag, **attrs):
    node = OxmlElement("w:"+tag)
    for key,value in attrs.items():
        node.set(qn("w:"+key),str(value))
    return node

def field(p, instruction, default="1"):
    p.add_run()._r.append(elem("fldChar",fldCharType="begin"))
    instr = elem("instrText")
    instr.set("{http://www.w3.org/XML/1998/namespace}space","preserve")
    instr.text = " "+instruction+" "
    p.add_run()._r.append(instr)
    p.add_run()._r.append(elem("fldChar",fldCharType="separate"))
    p.add_run(default)
    p.add_run()._r.append(elem("fldChar",fldCharType="end"))

def font(style, size, cn="宋体"):
    style.font.name="Times New Roman"
    style.font.size=Pt(size)
    style.font.color.rgb=RGBColor.from_string("19252F")
    style.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"),cn)

def build():
    path=ROOT/"research_report.docx"
    with tempfile.TemporaryDirectory() as tmp:
        raw=Path(tmp)/"pandoc.docx"
        pypandoc.convert_file(str(ROOT/"research_report.md"),"docx",outputfile=str(raw),
                              extra_args=["--resource-path="+str(ROOT)])
        doc=Document(raw)
    sec=doc.sections[0]
    sec.page_height,sec.page_width=Cm(29.7),Cm(21)
    sec.left_margin,sec.right_margin=Cm(2.25),Cm(2.25)
    sec.top_margin,sec.bottom_margin=Cm(2.2),Cm(2.05)
    font(doc.styles["Normal"],10.5)
    for name in ["Normal","Body Text","First Paragraph","Compact"]:
        style=doc.styles[name]
        font(style,10.5)
        style.paragraph_format.line_spacing=1.26
        style.paragraph_format.space_after=Pt(5)
    for name,size in [("Title",23),("Subtitle",13),("Heading 1",16),("Caption",9)]:
        style=doc.styles[name]
        font(style,size,"黑体" if name in ("Title","Heading 1") else "宋体")
    heading=doc.styles["Heading 1"]
    heading.paragraph_format.page_break_before=False
    heading.paragraph_format.space_before=Pt(16)
    heading.paragraph_format.space_after=Pt(10)
    # Native heading counters survive inserting and moving chapters.
    numbering=doc.part.numbering_part.element
    abstract_id=max([int(n.get(qn("w:abstractNumId"))) for n in numbering.findall(qn("w:abstractNum"))]+[-1])+1
    num_id=max([int(n.get(qn("w:numId"))) for n in numbering.findall(qn("w:num"))]+[0])+1
    abstract=elem("abstractNum",abstractNumId=abstract_id)
    abstract.append(elem("multiLevelType",val="singleLevel"))
    level=elem("lvl",ilvl=0)
    for tag,val in [("start","1"),("numFmt","decimal"),("pStyle",heading.style_id),("suff","space"),("lvlText","%1")]:
        level.append(elem(tag,val=val))
    abstract.append(level)
    numbering.insert(0,abstract)
    num=elem("num",numId=num_id);num.append(elem("abstractNumId",val=abstract_id));numbering.append(num)
    npr=heading.element.get_or_add_pPr().get_or_add_numPr()
    npr.get_or_add_ilvl().val=0;npr.get_or_add_numId().val=num_id
    header=sec.header.paragraphs[0]
    header.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    header.add_run("A股宽基指数多空对冲策略研究").font.size=Pt(9)
    footer=sec.footer.paragraphs[0]
    footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    field(footer,"PAGE")
    first=next(p for p in doc.paragraphs if p.style.name=="Heading 1")
    first.paragraph_format.page_break_before=True
    toc_title=first.insert_paragraph_before("目录")
    toc_title.runs[0].bold=True
    toc_title.runs[0].font.size=Pt(16)
    toc_title.paragraph_format.space_before=Pt(24)
    field(first.insert_paragraph_before(),r'TOC \o "1-1" \h \z \u',"打开 Word 后更新目录")
    for p in doc.paragraphs:
        if p.text.startswith(("表：","图：")):
            kind,caption=p.text.split("：",1)
            p.clear();p.style="Caption"
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.add_run(kind+" ");field(p,"SEQ "+("Table" if kind=="表" else "Figure")+r" \* ARABIC")
            p.add_run("　"+caption)
            p.paragraph_format.keep_with_next=kind=="表"
            p.paragraph_format.space_after=Pt(5)
        elif p._p.xpath(".//w:drawing"):
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next=True
    for table in doc.tables:
        borders=elem("tblBorders")
        for side in ("top","bottom","insideH"):
            borders.append(elem(side,val="single",sz="4",color="A5B3BD"))
        table._tbl.tblPr.append(borders)
        table.rows[0]._tr.get_or_add_trPr().append(elem("tblHeader",val="true"))
        for i,row in enumerate(table.rows):
            row._tr.get_or_add_trPr().append(elem("cantSplit"))
            for cell in row.cells:
                if i==0:
                    cell._tc.get_or_add_tcPr().append(elem("shd",fill="E8EFF3"))
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next=i<len(table.rows)-1
                    p.paragraph_format.space_after=Pt(2)
                    p.paragraph_format.space_before=Pt(2)
                    p.paragraph_format.line_spacing=1.05
                    for run in p.runs:
                        run.font.size=Pt(9)
                        run.bold=i==0
    for shape in doc.inline_shapes:
        if shape.width>Cm(16.3):
            shape.height=round(shape.height*Cm(16.3)/shape.width)
            shape.width=Cm(16.3)
    doc.save(path)
    return path

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--pdf",type=Path,help="Optional temporary pagination proof")
    parser.add_argument("--word-fields",action="store_true",help="Update page numbers and contents with desktop Word")
    args=parser.parse_args()
    path=build()
    if args.word_fields or args.pdf:
        import win32com.client
        word=win32com.client.DispatchEx("Word.Application")
        try:
            word.Visible=False; word.DisplayAlerts=0
            report=word.Documents.Open(str(path),AddToRecentFiles=False)
            report.Fields.Update()
            report.Repaginate()
            report.TablesOfContents(1).Update()
            report.Save()
            if args.pdf:
                report.ExportAsFixedFormat(str(args.pdf.resolve()),17)
            report.Close(False)
        finally:
            word.Quit()
