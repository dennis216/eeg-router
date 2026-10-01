"""Build a Chinese reproducibility dossier for the frozen H7 experiment."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, KeepTogether,
                                LongTable, PageBreak, PageTemplate, Paragraph,
                                Preformatted, Spacer, Table, TableStyle)

ROOT = Path('/root/b_false_alarm_atlas')
WIN_ROOT = '/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q'
OUT = Path(WIN_ROOT) / 'reports' / 'H7_dual_branch_recurrent_FP_suppressor_reproducibility_v1.pdf'
BASE = ROOT / 'outputs/reports'


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def pth(rel: str) -> Path:
    return BASE / rel


def sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def esc(value: object) -> str:
    return str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace('\n', '<br/>'), style)


def bullet(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph('&#8226; ' + text, style)


def make_table(data: list[list[object]], widths: list[float], font_size: float = 8.2,
               header: bool = True) -> Table:
    wrapped = []
    for ri, row in enumerate(data):
        wrapped.append([Paragraph(esc(cell), styles['TableHeader'] if header and ri == 0 else styles['TableCell']) for cell in row])
    table = LongTable(wrapped, colWidths=widths, repeatRows=1 if header else 0,
                      hAlign='LEFT', splitByRow=1)
    commands = [
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#C9D3DF')),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]
    if header:
        commands += [('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#17365D')),
                     ('TEXTCOLOR', (0, 0), (-1, 0), colors.white)]
        for r in range(1, len(data)):
            if r % 2 == 0:
                commands.append(('BACKGROUND', (0, r), (-1, r), colors.HexColor('#F3F6FA')))
    table.setStyle(TableStyle(commands))
    return table


def section(title: str) -> list:
    return [Spacer(1, 5), Paragraph(title, styles['H2']), HRFlowable(width='100%', thickness=0.7, color=colors.HexColor('#2F75B5')), Spacer(1, 4)]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('MSYH', 8)
    canvas.setFillColor(colors.HexColor('#607080'))
    canvas.drawString(18 * mm, 12 * mm, 'H7 reproducibility dossier | Frozen exploratory result')
    canvas.drawRightString(192 * mm, 12 * mm, f'Page {doc.page}')
    canvas.restoreState()


pdfmetrics.registerFont(TTFont('MSYH', '/mnt/c/Windows/Fonts/msyh.ttc'))
pdfmetrics.registerFont(TTFont('MSYH-Bold', '/mnt/c/Windows/Fonts/msyhbd.ttc'))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCN', parent=styles['Title'], fontName='MSYH-Bold', fontSize=21, leading=27, textColor=colors.HexColor('#17365D'), alignment=TA_CENTER, spaceAfter=10))
styles.add(ParagraphStyle(name='SubTitleCN', parent=styles['Normal'], fontName='MSYH', fontSize=10.5, leading=16, textColor=colors.HexColor('#4B5F73'), alignment=TA_CENTER, spaceAfter=10))
styles.add(ParagraphStyle(name='H2', parent=styles['Heading2'], fontName='MSYH-Bold', fontSize=13.5, leading=18, textColor=colors.HexColor('#17365D'), spaceBefore=6, spaceAfter=5))
styles.add(ParagraphStyle(name='H3CN', parent=styles['Heading3'], fontName='MSYH-Bold', fontSize=10.5, leading=14, textColor=colors.HexColor('#2F5597'), spaceBefore=5, spaceAfter=3))
styles.add(ParagraphStyle(name='BodyCN', parent=styles['BodyText'], fontName='MSYH', fontSize=9.2, leading=14.2, alignment=TA_LEFT, spaceAfter=5))
styles.add(ParagraphStyle(name='SmallCN', parent=styles['BodyText'], fontName='MSYH', fontSize=8.1, leading=11.3, textColor=colors.HexColor('#44515D'), spaceAfter=3))
styles.add(ParagraphStyle(name='TableCell', parent=styles['BodyText'], fontName='MSYH', fontSize=7.7, leading=10.2, wordWrap='CJK'))
styles.add(ParagraphStyle(name='TableHeader', parent=styles['BodyText'], fontName='MSYH-Bold', fontSize=7.8, leading=10.3, textColor=colors.white, wordWrap='CJK'))
styles.add(ParagraphStyle(name='CodeCN', parent=styles['Code'], fontName='Courier', fontSize=7.2, leading=9.5, leftIndent=7, rightIndent=7, backColor=colors.HexColor('#F4F6F8'), borderColor=colors.HexColor('#D5DDE5'), borderWidth=0.5, borderPadding=6, spaceAfter=6))
styles.add(ParagraphStyle(name='Callout', parent=styles['BodyText'], fontName='MSYH-Bold', fontSize=10.0, leading=15, textColor=colors.HexColor('#17365D'), backColor=colors.HexColor('#EAF2F8'), borderColor=colors.HexColor('#B4C7E7'), borderWidth=0.7, borderPadding=8, spaceBefore=3, spaceAfter=7))


def build_story() -> list:
    final = read_json(pth('h7-dual-branch-recurrent-fp-suppressor-v1-final-exploratory-freeze-v1/manifest.json'))
    policy = read_json(pth('h7-dual-branch-recurrent-fp-suppressor-v1-dev-policy-freeze-v1/manifest.json'))
    cnn = read_json(pth('h7-dual-branch-recurrent-fp-suppressor-v1-raw-cnn-label-support-audit-v1/manifest.json'))
    hold = read_json(pth('h7-dual-branch-recurrent-fp-suppressor-v1-exploratory-tusz-holdout-score-v1/manifest.json'))
    metrics = pd.read_csv(pth('h7-dual-branch-recurrent-fp-suppressor-v1-exploratory-tusz-holdout-score-v1/holdout_metrics.csv'))
    dev_curve = pd.read_csv(pth('h7-dual-branch-recurrent-fp-suppressor-v1-dev-template-safety-gate-v1/dev_method_gate_curve.csv'))
    fit_summary = pd.read_csv(pth('h7-dual-branch-recurrent-fp-suppressor-v1-eeg10-dual-template-fit-v1/template_summary.csv'))
    context = read_json(pth('h7-dual-branch-recurrent-fp-suppressor-v1-context-catalog-with-tn-tp-v1/manifest.json'))
    raw = read_json(pth('h7-dual-branch-recurrent-fp-suppressor-v1-raw-context-cache-v1/manifest.json'))
    feat = read_json(pth('h7-dual-branch-recurrent-fp-suppressor-v1-eeg10-with-tp-v1/manifest.json'))
    holdcat = read_json(pth('h7-dual-branch-recurrent-fp-suppressor-v1-exploratory-tusz-holdout-catalog-v1/manifest.json'))
    holdraw = read_json(pth('h7-dual-branch-recurrent-fp-suppressor-v1-exploratory-tusz-holdout-raw-v1/manifest.json'))
    holdeeg = read_json(pth('h7-dual-branch-recurrent-fp-suppressor-v1-exploratory-tusz-holdout-eeg10-v1/manifest.json'))

    story = []
    story += [Spacer(1, 22), Paragraph('H7 患者个体化 recurrent-FP template suppressor', styles['TitleCN']),
              Paragraph('方法、结果、结论、代码与完全复现说明 | v1 冻结探索性结果', styles['SubTitleCN']), Spacer(1, 8)]
    story.append(Paragraph('报告目的', styles['H2']))
    story.append(Paragraph('这份 PDF 只记录 H7 这一个实验：利用每位患者早期 EEG 中重复出现的 detector-specific false-positive pattern 和 shared false-positive pattern，构建低容量的患者个体化报警抑制器。它不是新的 seizure detector，也不是在 SingLEM、CBraMod、TCN-GAT 之间选择模型的 router。', styles['BodyCN']))
    story.append(Paragraph('最终状态：模板方法在 development 上通过安全 gate，但在 TUSZ 后期 chronological holdout 上没有安全泛化。因此结果已冻结为 exploratory negative result；没有训练 raw CNN，也没有根据 holdout 调参。', styles['Callout']))

    story += section('1. 研究问题与预先规定的假设')
    story.append(Paragraph('<b>问题：</b>某位患者早期出现的、可重复的 detector false-positive EEG pattern，能否在该患者后续记录中再次识别，并减少 false alarms，同时不明显压低 seizure sensitivity？', styles['BodyCN']))
    story.append(Paragraph('<b>假设：</b>患者内的重复 FP morphology 可能比跨患者共享规律更稳定；若模板在 development 中覆盖可抑制 FP 且不覆盖 seizure，则可作为报警后处理 suppressor。', styles['BodyCN']))
    story.append(Paragraph('<b>核心边界：</b>“像历史 FP”只表示统计相似，不表示已证明眼动、肌电或电极机制，也不表示 FP 的生理原因已经被识别。', styles['BodyCN']))

    story += section('2. 数据、模型与时间划分')
    data_rows = [
        ['项目', '本实验冻结定义'],
        ['数据', 'TUSZ v2.0.6；Windows: D:\\TUH_EEG\\TUSZ_v2.0.6；WSL: /mnt/d/TUH_EEG/TUSZ_v2.0.6'],
        ['experts', 'SingLEM、CBraMod、TCN-GAT；三个 seizure detector 全程冻结，使用其已有概率和事件化结果'],
        ['患者时间切分', '每位患者按完整 EDF 的真实时间顺序：最早 60% fit，中间 20% dev，最后 20% chronological holdout'],
        ['fit 资格', '每个 patient-detector 单元至少 20 个 fit FP、跨至少 3 条 EDF，并且 fit 中至少 3 个原始 seizure annotation'],
        ['资格规模', f"72 个 patient-detector 单元，40 位患者；上下文目录输出为 {context.get('total_rows', context.get('contexts', 'N/A'))} 行"],
        ['holdout 语义', 'TUSZ chronological holdout 以前曾在旧 H7 工作中被查看，所以本次只可标为探索性，不能作为独立 confirmatory result'],
    ]
    story.append(make_table(data_rows, [35*mm, 145*mm]))
    story.append(Paragraph('每个窗口都绑定 patient ID 和完整 EDF；患者的 EDF、事件和窗口不会跨 fit、dev、holdout 混放。三种 expert 不在目标患者上重新训练，避免把 expert adaptation 和 suppressor 效果混在一起。', styles['BodyCN']))

    story += section('3. 标签与控制组定义')
    label_rows = [
        ['标签/对象', '定义', '用途'],
        ['FP_m', 'detector m 在无 seizure 区间产生的冻结报警事件', 'specific template 的候选正例'],
        ['TN_m', '同患者、同 EDF、同 fit/dev 阶段、同记录时间三分位的 20 秒背景窗口；排除 seizure 与该 detector 报警附近 30 秒', '防止模板只学习患者背景'],
        ['TP_m', '冻结事件表中 is_fp=0 的 seizure-matched 报警，fit/dev only', 'seizure hard negative，检查 suppressor 是否压掉真实检出'],
        ['shared FP candidate', '至少两个 detector 在同一记录内的 FP onset 相差不超过 +/-10 秒；只作为额外候选标志', 'shared branch 的 fit 正例候选；不要求部署时当前也有两个模型同时报警'],
        ['unroutable / all-miss', '三个 detector 都漏掉真实 seizure', '不伪造“最佳 expert”，并作为安全边界'],
    ]
    story.append(make_table(label_rows, [32*mm, 95*mm, 53*mm]))
    story.append(Paragraph('shared FP 的 +/-10 秒只是本实验的操作性 onset collar，不是 EEG 领域唯一标准；它用于定义候选关联，不能单独推出共同 artifact 或共同机制。', styles['SmallCN']))

    story += section('4. Raw EEG 预处理与 EEG10 表示')
    story.append(Paragraph('每个 FP、TN 或 TP 上下文都从同一份原始 EDF 提取，先转换到统一的 16 通道 bipolar montage，再重采样到 128 Hz。上下文是事件 onset 结束的 20 秒 causal window，因此只使用报警时刻之前已发生的信号。原始窗口形状为 16 x 2560，并以 float16 缓存。', styles['BodyCN']))
    story.append(Paragraph('之后将 20 秒切成十个不重叠的 2 秒 frame，计算 10 个 EEG-only 特征：', styles['BodyCN']))
    feat_rows = [
        ['类别', '特征'],
        ['振幅', 'RMS median、RMS IQR'],
        ['空间', 'log-RMS spatial dispersion、log-RMS max-min'],
        ['频带', 'delta、theta、alpha、beta、gamma relative power'],
        ['比值', '(beta + gamma) / (delta + theta) high/low ratio'],
    ]
    story.append(make_table(feat_rows, [35*mm, 145*mm]))
    story.append(Paragraph('没有把 detector probability、model ID、HMM state、event duration 或人工 artifact label 放入输入特征。scaler 只在 fit 数据上拟合；dev 和 holdout 只做 transform。', styles['BodyCN']))

    story += section('5. Template discovery 方法')
    story.append(Paragraph('<b>5.1 Detector-specific branch：</b>对每一个合格 patient-detector 单元，取 fit FP_m 和 matched TN_m；用 fit-only StandardScaler 标准化；对每个候选距离 cutoff 做 leave-one-EDF-out fit 评估：把一条 EDF 的 FP/TN 留出，用其他 EDF 的 FP 作为参考，计算最近 FP 距离并选择 balanced accuracy 最好的 cutoff。', styles['BodyCN']))
    story.append(Paragraph('在冻结 cutoff 下，把 fit FP 之间的距离构成图：只有跨不同 EDF 的边才允许连接。连通分量必须至少包含 3 个 FP event 且来自至少 2 条 EDF，才成为 recurrent specific template。每个分量用其标准化特征的中位数作为 prototype。', styles['BodyCN']))
    story.append(Paragraph('<b>5.2 Shared branch：</b>在患者内用 fit shared-FP candidate 去重后的上下文建立模板；使用相同的跨 EDF recurrence 和 fit-only cutoff 规则。它是患者级 shared-FP pattern，而不是固定的生理 artifact 类别。', styles['BodyCN']))
    story.append(Paragraph(f"实际 fit 输出：specific branch 共 {int(fit_summary[fit_summary.branch.eq('specific')].components.sum())} 个 component、{int(fit_summary[fit_summary.branch.eq('specific')].template_events.sum())} 个模板事件；shared branch 共 {int(fit_summary[fit_summary.branch.eq('shared')].components.sum())} 个 component、{int(fit_summary[fit_summary.branch.eq('shared')].template_events.sum())} 个模板事件。", styles['BodyCN']))

    story += section('6. Development 安全 gate')
    story.append(Paragraph('模板在 fit 中冻结后，dev 只用于选择一个全局距离倍率 alpha。预先固定 alpha 网格为 0.25、0.50、0.75、1.00、1.25、1.50、2.00。每个策略选择在 dev 上 FP-event removal 最大、且相对冻结 detector 的 matched-event sensitivity loss 不超过 3 个百分点的 alpha。', styles['BodyCN']))
    dsel = dev_curve[dev_curve.selected]
    dev_rows = [['策略', 'alpha', 'dev FP 基线', '移除 FP', 'FP 减少', 'TP 损失', '敏感度损失', '是否安全']]
    for _, r in dsel.iterrows():
        dev_rows.append([r['method'], f"{r['alpha']:.2f}", int(r['baseline_fp_events']), int(r['suppressed_fp_events']), f"{r['fa_reduction_fraction']*100:.1f}%", int(r['suppressed_matched_events']), f"{r['sensitivity_loss_pp']:.2f} pp", '是'])
    story.append(make_table(dev_rows, [25*mm, 18*mm, 24*mm, 22*mm, 23*mm, 22*mm, 28*mm, 18*mm]))
    story.append(Paragraph('development 主规则冻结为 shared alpha=1.50；specific alpha=1.25、dual-OR alpha=1.00 作为预先保留的消融。shared 在 dev 中表现最好，但这不允许用 holdout 再挑一次。', styles['Callout']))

    story += section('7. Holdout 评分规则')
    story.append(Paragraph('holdout 只读取已冻结模板、scaler、cutoff 和 alpha。对每个冻结 detector event，如果距离 ratio 小于对应 alpha，则标记为 suppressed：specific 只看 detector-specific 分支；shared 只看 shared 分支；dual-OR 满足任一分支即可。没有在 holdout 重新拟合或修改任何参数。', styles['BodyCN']))
    story.append(Paragraph('事件级指标包括 pooled sensitivity、FA/24h 和患者级 waterfall；另保存 2,000 次患者 bootstrap。由于本次 TUSZ holdout 在更早的 H7 工作中曾被查看，所有结果均标为 exploratory，不能被称为独立外部确认。', styles['BodyCN']))

    story += section('8. 实际结果')
    result_rows = [['方法', '患者单元', '有 holdout seizure 患者', 'pooled sensitivity', 'macro sensitivity', 'pooled FA/24h', 'macro FA/24h']]
    for _, r in metrics.iterrows():
        result_rows.append([r['method'], int(r['patients']), int(r['patients_with_truth']), f"{r['pooled_sensitivity']*100:.1f}%", f"{r['macro_sensitivity']*100:.1f}%", f"{r['pooled_FA_24h']:.1f}", f"{r['macro_FA_24h']:.1f}"])
    story.append(make_table(result_rows, [25*mm, 23*mm, 28*mm, 30*mm, 30*mm, 25*mm, 25*mm]))
    story.append(Paragraph('holdout 基线为 pooled sensitivity 33.8%、FA/24h 110.4。specific 降至 29.2% sensitivity 和 99.2 FA/24h；shared 降至 22.5% 和 92.8；dual-OR 降至 28.8% 和 99.2。三种策略虽然降低了 FA/24h，但都牺牲了过多 seizure sensitivity。', styles['BodyCN']))
    story.append(Paragraph('相对 baseline 的 pooled 变化：specific sensitivity -4.6 pp、FA/24h -11.2；shared sensitivity -11.3 pp、FA/24h -17.6；dual-OR sensitivity -5.0 pp、FA/24h -11.2。', styles['Callout']))

    story += section('9. CNN gate 与最终结论')
    story.append(Paragraph(f"raw CNN 没有启动。预先门槛为每个 patient-local branch 至少 30 个正例、30 个负例、正例跨至少 3 条 EDF；audit 结果是 specific {cnn['counts'][1]['supported'] if len(cnn.get('counts', [])) > 1 else 0}/72 单元达标，shared 仅 3/40 患者达标。以当前样本量训练 CNN 会把早期记录细节当作规则，无法证明后期泛化。", styles['BodyCN']))
    story.append(Paragraph('<b>结论：</b>早期患者内 recurrent-FP template 在 development 中能覆盖一部分误报，但这种收益没有安全地迁移到后期 TUSZ 记录。当前证据不支持继续训练 CNN，也不支持把该 suppressor 宣称为可用的报警抑制政策。', styles['Callout']))
    story.append(Paragraph('<b>可以声称：</b>患者早期重复 FP pattern 在同一患者的 development 数据上可被低容量模板识别；这种 pattern 在后期 TUSZ 中出现明显 temporal drift 或不稳定性。<br/><b>不能声称：</b>找到了共同生理机制；模板代表眼动、肌电或电极 artifact；抑制器已经提高临床性能；TUSZ holdout 是独立确认。', styles['BodyCN']))

    story += section('10. 复现所需的冻结输出')
    output_rows = [['阶段', 'WSL 输出目录', '关键文件/状态']]
    outputs = [
        ('fit-only eligibility', 'h7-dual-branch-recurrent-fp-suppressor-v1-fit-only-eligibility-v2', 'patient_detector_fit_only_eligibility.csv; 72 units / 40 patients'),
        ('context catalog', 'h7-dual-branch-recurrent-fp-suppressor-v1-context-catalog-with-tn-tp-v1', 'context_catalog_with_tn_tp.parquet; 9,533 rows'),
        ('raw cache', 'h7-dual-branch-recurrent-fp-suppressor-v1-raw-context-cache-v1', 'manifest.json; 7,156 / 7,159 contexts; 3 frozen exclusions'),
        ('EEG10', 'h7-dual-branch-recurrent-fp-suppressor-v1-eeg10-with-tp-v1', 'context_eeg10_with_tp.parquet; 7,636 unique contexts'),
        ('template fit', 'h7-dual-branch-recurrent-fp-suppressor-v1-eeg10-dual-template-fit-v1', 'objects/*.joblib; fit_dev_template_scores.parquet; 134 specific / 50 shared components'),
        ('dev gate', 'h7-dual-branch-recurrent-fp-suppressor-v1-dev-template-safety-gate-v1', 'dev_method_gate_curve.csv; dev_patient_unit_action_curve.csv'),
        ('policy freeze', 'h7-dual-branch-recurrent-fp-suppressor-v1-dev-policy-freeze-v1', 'manifest.json; primary shared alpha=1.5'),
        ('exploratory holdout', 'h7-dual-branch-recurrent-fp-suppressor-v1-exploratory-tusz-holdout-score-v1', 'holdout_metrics.csv; patient_waterfall.csv; patient_bootstrap_delta.csv'),
        ('final freeze', 'h7-dual-branch-recurrent-fp-suppressor-v1-final-exploratory-freeze-v1', 'manifest.json; conclusion.txt'),
    ]
    for row in outputs:
        output_rows.append([row[0], f'/root/b_false_alarm_atlas/outputs/reports/{row[1]}', row[2]])
    story.append(make_table(output_rows, [30*mm, 85*mm, 55*mm]))

    story += section('11. 代码清单与作用')
    code_rows = [['脚本', '作用']]
    scripts = [
        ('110_preflight_no_duplicate_overlap.py', '每个新阶段前检查重复、checkpoint reuse、协议重叠和活动进程'),
        ('177_freeze_h7_dual_branch_protocol.py', '初版双分支协议冻结'),
        ('179_freeze_h7_dual_branch_standardized_addendum.py', '行业标准化 addendum，确定 chronology、fit eligibility、controls、指标'),
        ('180_h7_fit_only_annotation_eligibility.py', '使用原始 TUSZ csv_bi seizure annotations 审计 fit-only 资格'),
        ('181_h7_dual_branch_context_catalog.py', '建立 fit/dev FP 与 seizure safety context catalog'),
        ('182_h7_add_matched_tn_contexts.py', '同患者、同 EDF、同时间分层 deterministic TN 控制'),
        ('186_h7_add_tp_contexts.py', '加入 fit/dev detector TP safety contexts'),
        ('183_h7_materialize_raw_context_cache.py', '从 EDF 提取 20 秒 raw context，16 channel、128 Hz'),
        ('184_h7_freeze_raw_cache_exclusions.py', '冻结 3 个 EDF 尾部越界窗口的排除审计'),
        ('185_h7_extract_cached_eeg10.py', 'FP/TN/seizure raw cache 转 EEG10'),
        ('188_h7_extract_eeg10_with_tp.py', 'FP/TN/TP 完整 catalog 转 EEG10'),
        ('186_h7_fit_eeg10_dual_templates.py', 'fit-only specific/shared template 与 LOEO cutoff'),
        ('189_h7_score_templates_and_dev_safety_gate.py', 'dev template scoring、alpha gate、安全性曲线'),
        ('190_h7_raw_cnn_label_support_audit.py', 'CNN 正例/负例/跨 EDF 支持量审计，不训练'),
        ('191_h7_freeze_dev_selected_template_policy.py', '冻结 dev 选择的 primary 与 secondary policy'),
        ('192_h7_build_exploratory_holdout_event_catalog.py', '建立 holdout FP/TP catalog，标记 exploratory'),
        ('193_h7_materialize_exploratory_holdout_raw.py', '提取 holdout raw context'),
        ('194_h7_extract_exploratory_holdout_eeg10.py', 'holdout raw context 转 EEG10'),
        ('195_h7_score_exploratory_holdout_policies.py', '一次性应用冻结策略并计算 pooled/macro/bootstrap'),
        ('196_h7_freeze_exploratory_negative_result.py', '冻结最终探索性阴性结果'),
    ]
    for script, desc in scripts:
        code_rows.append([f'scripts/{script}', desc])
    story.append(make_table(code_rows, [78*mm, 102*mm]))
    story.append(Paragraph('Windows 工作区代码目录：C:\\Users\\User\\Documents\\Codex\\2026-08-03\\du-q\\scripts\\。WSL 实际运行目录：/root/b_false_alarm_atlas/scripts/。代码文件在复制到 WSL 后才是实际运行版本；每次脚本均使用新的 immutable output namespace 并拒绝覆盖已有 manifest。', styles['BodyCN']))

    story += section('12. 复现顺序与命令')
    story.append(Paragraph('下面命令描述原始执行顺序。冻结输出已经存在时，脚本会拒绝覆盖；要做完全独立复现，应先复制整个 repository 和输入数据到干净位置，或为每一步创建新 namespace。不要在当前冻结目录上删除旧结果。', styles['BodyCN']))
    commands = '''wsl -d Ubuntu-24.04 --user root
cd /root/b_false_alarm_atlas
source .venv/bin/activate

# 1. Run the no-duplicate preflight before each new stage
python scripts/110_preflight_no_duplicate_overlap.py --namespace <new-namespace>

# 2. Build the fit-only catalog and safety controls
python scripts/180_h7_fit_only_annotation_eligibility.py
python scripts/181_h7_dual_branch_context_catalog.py
python scripts/182_h7_add_matched_tn_contexts.py
python scripts/186_h7_add_tp_contexts.py

# 3. Materialize raw EEG and derive EEG10
python scripts/183_h7_materialize_raw_context_cache.py
python scripts/184_h7_freeze_raw_cache_exclusions.py
python scripts/185_h7_extract_cached_eeg10.py
python scripts/188_h7_extract_eeg10_with_tp.py

# 4. Fit templates, freeze the dev gate, and audit CNN support
python scripts/186_h7_fit_eeg10_dual_templates.py
python scripts/189_h7_score_templates_and_dev_safety_gate.py
python scripts/190_h7_raw_cnn_label_support_audit.py
python scripts/191_h7_freeze_dev_selected_template_policy.py

# 5. Score the TUSZ chronological holdout as exploratory only
python scripts/192_h7_build_exploratory_holdout_event_catalog.py
python scripts/193_h7_materialize_exploratory_holdout_raw.py
python scripts/194_h7_extract_exploratory_holdout_eeg10.py
python scripts/195_h7_score_exploratory_holdout_policies.py
python scripts/196_h7_freeze_exploratory_negative_result.py'''
    story.append(Preformatted(commands, styles['CodeCN']))
    story.append(Paragraph('依赖：Python 3.11、numpy、pandas、scipy、scikit-learn、joblib、pyarrow、mne、reportlab。原始数据和冻结 expert event parquet 必须与 manifest 中的 source hash 一致。', styles['SmallCN']))

    story += section('13. 复现审计与下一步')
    hash_rows = [['冻结对象', 'SHA-256']]
    for label, rel in [
        ('final freeze manifest', 'h7-dual-branch-recurrent-fp-suppressor-v1-final-exploratory-freeze-v1/manifest.json'),
        ('dev policy manifest', 'h7-dual-branch-recurrent-fp-suppressor-v1-dev-policy-freeze-v1/manifest.json'),
        ('holdout score manifest', 'h7-dual-branch-recurrent-fp-suppressor-v1-exploratory-tusz-holdout-score-v1/manifest.json'),
        ('template fit manifest', 'h7-dual-branch-recurrent-fp-suppressor-v1-eeg10-dual-template-fit-v1/manifest.json'),
        ('raw CNN audit manifest', 'h7-dual-branch-recurrent-fp-suppressor-v1-raw-cnn-label-support-audit-v1/manifest.json'),
    ]:
        path = pth(rel)
        hash_rows.append([label, sha(path) if path.exists() else 'missing'])
    story.append(make_table(hash_rows, [65*mm, 115*mm]))
    story.append(Paragraph('下一步不能是继续用同一 TUSZ holdout 调整模板。若要得到 confirmatory 结论，需要一套本次研究前从未接触过的外部患者队列，并在运行前冻结同一 60/20/20 chronology、模板规则、alpha、指标和安全 gate。若没有新队列，最合理做法是把本实验作为 False-Alarm Atlas 的患者个体化抑制阴性边界。', styles['BodyCN']))
    return story


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame = Frame(18*mm, 18*mm, 174*mm, 257*mm, id='normal', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=17*mm, bottomMargin=18*mm, title='H7 reproducibility dossier')
    doc.addPageTemplates([PageTemplate(id='main', frames=frame, onPage=footer)])
    doc.build(build_story())
    print(OUT)


if __name__ == '__main__':
    main()
