"""Local chapter assets and copyable LaTeX blocks; never edit a manuscript."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

REPO = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO/'img'
CHAPTERS = ('03_methodology','04_results')


def chapter_paths(output_root, chapter):
    if chapter not in CHAPTERS:
        raise ValueError(f'Unknown chapter: {chapter}')
    folder = Path(output_root).resolve()/chapter
    data = folder/'data'
    data.mkdir(parents=True,exist_ok=True)
    return folder,data


def tex_escape(value):
    replacements = {'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
                    '#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(replacements.get(char,char) for char in str(value))


def table_block(headers, rows, caption, label):
    """Return a complete table environment; headers/caption may contain LaTeX."""
    lines = [r'\begin{table}[htbp]',r'\centering',r'\caption{'+caption+'}',
             r'\label{'+label+'}',r'\small',r'\begin{tabular}{'+'r'*len(headers)+'}',
             r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
    lines.extend(' & '.join(str(value) for value in row)+r' \\' for row in rows)
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}',r'\end{table}'])+'\n'


def write_tables(folder, tables, template):
    folder = Path(folder);snippets = folder/'snippets';snippets.mkdir(exist_ok=True)
    blocks=[]
    for item in tables:
        block = table_block(item['headers'],item['rows'],item['caption'],item['label'])
        (snippets/(item['stem']+'.tex')).write_text(block,encoding='utf-8')
        blocks.append(block)
    (folder/'tables_all.tex').write_text('% Requires booktabs. Copy selected table blocks into the main text or appendix.\n\n'+'\n'.join(blocks),encoding='utf-8')
    empty = table_block(template['headers'],[['']*len(template['headers'])],template['caption'],template['label'])
    (folder/'table_template.tex').write_text('% Empty table: replace caption/label and add rows. Requires booktabs.\n'+empty,encoding='utf-8')
    (folder/'table_manifest.json').write_text(json.dumps([
        {k:v for k,v in item.items() if k!='rows'} | dict(rows=len(item['rows'])) for item in tables],indent=2),encoding='utf-8')


def export_chapter(folder, figures, pdf_name, chapter, title, mode):
    """Export complete figure blocks, gallery and an independent preview document."""
    folder=Path(folder);snippets=folder/'snippets';snippets.mkdir(exist_ok=True)
    groups={'all':[],'main':[],'appendix':[]}
    for figure in figures:
        stem=figure['stem']
        if not (folder/(stem+'.pdf')).is_file():
            raise FileNotFoundError(f'Missing standalone PDF; regenerate this figure: {stem}')
        label='fig:'+chapter+'_'+re.sub('[^A-Za-z0-9]+','_',stem)
        block='\n'.join([r'\begin{figure}[htbp]',r'\centering',
            r'\includegraphics[width='+f"{figure['width_mm']:.4f}"+r'mm]{img/'+chapter+'/'+stem+r'.pdf}',
            r'\caption{'+figure['caption']+'}',r'\label{'+label+'}',r'\end{figure}'])+'\n'
        (snippets/(stem+'.tex')).write_text(block,encoding='utf-8')
        groups['all'].append(block);groups['main' if figure['main'] else 'appendix'].append(block)
    for name,blocks in groups.items():
        (folder/('figures_'+name+'.tex')).write_text('% Copy selected complete figure environments; placement is deferred. Requires graphicx.\n\n'+'\n'.join(blocks),encoding='utf-8')
    (folder/'figure_manifest.json').write_text(json.dumps(figures,indent=2),encoding='utf-8')
    gallery=['<!doctype html><meta charset="utf-8"><title>'+html.escape(title)+'</title>',
        '<style>body{max-width:1100px;margin:2rem auto;font:17px Georgia}img{max-width:100%;height:auto}article{margin:3rem 0}</style>',
        '<h1>'+html.escape(title)+'</h1><p>Mode: '+html.escape(mode)+'. Main/appendix tags are suggestions; no manuscript was edited.</p>',
        '<p><a href="'+pdf_name+'">Combined PDF</a> · <a href="figures_all.tex">Figure LaTeX</a> · <a href="tables_all.tex">Table LaTeX</a></p>']
    for item in figures:
        stem=html.escape(item['stem'])
        gallery.append(f'<article><h2>{stem}</h2><a href="{stem}.pdf"><img src="{stem}.png"></a><p>{html.escape(item["caption"])}</p></article>')
    (folder/'index.html').write_text('\n'.join(gallery),encoding='utf-8')
    preview='\n'.join([r'\documentclass[12pt]{article}',r'\usepackage[a4paper,margin=20mm]{geometry}',
        r'\usepackage{graphicx,booktabs,amsmath}',r'\begin{document}',
        r'\input{img/'+chapter+r'/figures_all.tex}',r'\clearpage',
        r'\input{img/'+chapter+r'/tables_all.tex}',r'\end{document}'])+'\n'
    (folder/'preview.tex').write_text(preview,encoding='utf-8')
    (folder/'README.md').write_text(f'# {title}\n\nMode: `{mode}`. Figures: {len(figures)}.\n\n'
        'PDF/SVG/450-dpi PNG figures and complete LaTeX figure/table blocks are generated here. '
        'The snippets reference `img/'+chapter+'/`; copy the asset folder with the snippets when later moving to a manuscript. '
        'Main/appendix groupings are suggestions, not automatic placement. No manuscript prose or files are changed.\n\n'
        '`data/` contains numerical inputs/tables and provenance. Keep these separate from visual-only refreshes. '
        '`table_template.tex` has an empty table. `snippets/` contains individual blocks. '
        '`preview.tex` can be compiled from the repository root; it does not modify the thesis.\n',encoding='utf-8')


def import_regeneration_inputs(archive, output_root):
    """Restore whitelisted numerical inputs from the cleanup ZIP into local chapters."""
    archive=Path(archive).resolve();records=[]
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            parts=PurePosixPath(member.filename).parts
            if member.is_dir():continue
            if (any(part in ('.','..') or '\\' in part or ':' in part for part in parts)
                    or PurePosixPath(member.filename).is_absolute()):
                raise ValueError('Unsafe archive path: '+member.filename)
            destination=None
            if len(parts)>=3 and parts[:2]==('tab_new','results_plan') and parts[-1].endswith('.csv'):
                destination=chapter_paths(output_root,'04_results')[1].joinpath(*parts[2:])
            elif len(parts)>=3 and parts[:2]==('tab_new','methodology_plan') and parts[-1].endswith('.csv'):
                destination=chapter_paths(output_root,'03_methodology')[1].joinpath(*parts[2:])
            elif len(parts)>=3 and parts[0]=='new_pic' and parts[1] in ('data','tables') and Path(parts[-1]).suffix in ('.csv','.json','.npz'):
                destination=chapter_paths(output_root,'04_results')[1].joinpath('reference',*parts[1:])
            if destination is None:continue
            content=source.read(member)
            if destination.exists() and destination.read_bytes()!=content:
                raise ValueError('Refusing to overwrite different imported input: '+str(destination))
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(content)
            records.append(dict(archive_member=member.filename,path=str(destination),sha256=hashlib.sha256(content).hexdigest()))
    if not records:raise ValueError('No recognized numerical inputs in archive.')
    for chapter in CHAPTERS:
        folder,data=chapter_paths(output_root,chapter)
        (data/'import_manifest.json').write_text(json.dumps(dict(archive=str(archive),
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=[r for r in records if Path(r['path']).is_relative_to(folder)]),indent=2),encoding='utf-8')
    return records
