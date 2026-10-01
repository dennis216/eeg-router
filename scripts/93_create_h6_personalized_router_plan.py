#!/usr/bin/env python3
"""Create a detailed PDF plan for supervised within-patient phenotype routing."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path("/root/b_false_alarm_atlas/outputs/reports/v3-groupkfold-confirmatory-v1/h6_supervised_personalized_router_experimental_plan.pdf")
FONT_PATH = Path("/mnt/c/Windows/Fonts/msyh.ttc")
if FONT_PATH.exists():
    FONT = FontProperties(fname=str(FONT_PATH))
else:
    FONT = FontProperties(family="DejaVu Sans")


def add_text(fig, x, y, text, size=10, weight="normal", color="#222222", ha="left", va="top", **kwargs):
    fig.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        fontproperties=FONT,
        linespacing=1.35,
        **kwargs,
    )


def page_base(title, number, subtitle=None):
    fig = plt.figure(figsize=(8.5, 11))
    add_text(fig, 0.06, 0.965, title, size=18, weight="bold", color="#183A61")
    if subtitle:
        add_text(fig, 0.06, 0.932, subtitle, size=9, color="#555555")
    add_text(fig, 0.94, 0.025, f"{number} / 7", size=8, color="#777777", ha="right")
    return fig


def boxed_text(fig, x, y, w, h, title, body, face="#F4F7FB", edge="#7A9CC6", title_color="#183A61"):
    patch = FancyBboxPatch(
        (x, y - h),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
        transform=fig.transFigure,
    )
    fig.patches.append(patch)
    add_text(fig, x + 0.018, y - 0.022, title, size=11, weight="bold", color=title_color)
    add_text(fig, x + 0.018, y - 0.065, body, size=8.7, color="#222222")


def draw_flow(fig, boxes, y=0.63):
    x_positions = [0.07, 0.29, 0.51, 0.73]
    for index, (title, body, color) in enumerate(boxes):
        x = x_positions[index]
        patch = FancyBboxPatch(
            (x, y - 0.18),
            0.17,
            0.18,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=1.2,
            edgecolor=color,
            facecolor="#F7FAFE",
            transform=fig.transFigure,
        )
        fig.patches.append(patch)
        add_text(fig, x + 0.085, y - 0.035, title, size=9.5, weight="bold", color=color, ha="center")
        add_text(fig, x + 0.085, y - 0.088, body, size=7.8, ha="center")
        if index < len(boxes) - 1:
            arrow = FancyArrowPatch(
                (x + 0.175, y - 0.09),
                (x_positions[index + 1] - 0.008, y - 0.09),
                arrowstyle="->",
                mutation_scale=13,
                linewidth=1.3,
                color="#6B7280",
                transform=fig.transFigure,
            )
            fig.patches.append(arrow)


def style_table(table, header_color="#2F5597", body_color="#FFFFFF", alt_color="#F3F6FA", fontsize=8.2):
    for (row, col), cell in table.get_celld().items():
        cell.get_text().set_fontproperties(FONT)
        cell.get_text().set_fontsize(fontsize)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(body_color if row % 2 else alt_color)
        cell.set_edgecolor("#B7C9E2")
        cell.set_linewidth(0.5)


def make_pdf():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(OUT)) as pdf:
        # Page 1
        fig = page_base(
            "H6: 有监督的患者个性化 Phenotype-Router 实验计划",
            1,
            "h6-supervised-personalized-router-v1 | 研究计划，不修改已完成的标准 v3 结果",
        )
        add_text(fig, 0.06, 0.855, "核心问题", size=13, weight="bold", color="#183A61")
        add_text(
            fig,
            0.06,
            0.818,
            "如果先用同一个患者前半段的有标签 EEG 学习该患者的 phenotype 和模型适配关系，\n"
            "能否让后半段的 seizure 检测比固定使用一个全局模型更有效？",
            size=12,
        )
        boxed_text(
            fig,
            0.06,
            0.675,
            0.41,
            0.17,
            "这是什么",
            "一个按时间顺序进行的患者自适应实验。\n前半段用于校准，后半段是冻结后的 holdout。",
        )
        boxed_text(
            fig,
            0.53,
            0.675,
            0.41,
            0.17,
            "这不是什么",
            "不是重新训练 SingLEM、CBraMod 或 TCN-GAT，\n也不是重新调当前 TUSZ test 阈值。",
            face="#FFF8EE",
            edge="#D79A3B",
            title_color="#8A5A00",
        )
        add_text(fig, 0.06, 0.43, "预期贡献", size=13, weight="bold", color="#183A61")
        add_text(
            fig,
            0.08,
            0.395,
            "1. 检验患者早期标签是否能帮助后期选择更合适的模型。\n"
            "2. 区分一般的 ensemble/router 效果和真正的患者个性化效果。\n"
            "3. 给出每位患者的适应收益、fallback 情况和失败原因。\n"
            "4. 将患者无关泛化和患者内个性化分开报告。",
            size=10,
        )
        add_text(fig, 0.06, 0.205, "推荐的主要结论门槛", size=13, weight="bold", color="#183A61")
        boxed_text(
            fig,
            0.06,
            0.17,
            0.88,
            0.105,
            "Primary success gate",
            "相对于全局最佳单模型，后半段 FA/24h 降低至少 25%，\n"
            "同时 sensitivity 损失不超过 3 个百分点；\n"
            "否则报告为 not supported，不根据 holdout 调整规则。",
            face="#EEF7EE",
            edge="#6BA56B",
            title_color="#2F6B2F",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2
        fig = page_base("1. 数据划分和时间顺序", 2, "患者内迁移必须与患者无关泛化分开")
        add_text(fig, 0.06, 0.875, "每个标准外层 test 患者内部再做一次 chronological split", size=13, weight="bold", color="#183A61")
        draw_flow(
            fig,
            [
                ("患者全集", "标准 GroupKFold\n外层 test 患者", "#315A8A"),
                ("前 50%", "calibration\n有标签", "#2F7D54"),
                ("冻结规则", "phenotype -> model\n只在此处学习", "#8A5A00"),
                ("后 50%", "chronological\nholdout", "#9B3D3D"),
            ],
            y=0.75,
        )
        add_text(fig, 0.06, 0.49, "划分规则", size=13, weight="bold", color="#183A61")
        add_text(
            fig,
            0.08,
            0.455,
            "• 按 recording 的时间顺序划分，而不是随机切 EEG window。\n"
            "• 一个完整 EDF 只能进入 calibration 或 holdout 的一边。\n"
            "• 前后边界留出 30 秒缓冲，避免同一事件跨越边界。\n"
            "• 后半段标签只用于最后评分，不能参与 phenotype、模型或阈值选择。\n"
            "• 三个 backbone、CHB validation-only threshold 和 checkpoint 全部冻结。",
            size=9.5,
        )
        boxed_text(
            fig,
            0.06,
            0.255,
            0.88,
            0.16,
            "重要解释",
            "同一个患者同时出现在前半段和后半段并不是泄漏，\n"
            "因为这是有意测试的患者内适应场景。\n"
            "但是它不能被称为新的患者无关泛化证据，\n"
            "必须标记为 within-patient personalization。",
            face="#FFF8EE",
            edge="#D79A3B",
            title_color="#8A5A00",
        )
        add_text(fig, 0.06, 0.12, "建议主分析单位：患者；seed 作为重复运行因素；不要把 window 数当作独立样本数。", size=9, color="#555555")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3
        fig = page_base("2. Phenotype 和个性化 Router 如何工作", 3, "先学习患者特征与模型表现的关系，再冻结到后半段")
        draw_flow(
            fig,
            [
                ("前半段 EEG", "信号特征 +\n三个模型概率", "#315A8A"),
                ("Phenotype", "分配到冻结\nphenotype 原型", "#6A4C93"),
                ("患者校准表", "每种 phenotype\n比较三个模型", "#2F7D54"),
                ("后半段", "按 phenotype\n选择模型", "#9B3D3D"),
            ],
            y=0.82,
        )
        add_text(fig, 0.06, 0.56, "Phenotype 的来源", size=13, weight="bold", color="#183A61")
        add_text(
            fig,
            0.08,
            0.525,
            "使用已有的事件级特征：事件持续时间、peak probability、RMS、频谱、空间差异、"
            "line-noise、clipping 和 flatness 等。\n"
            "scaler、PCA 和 phenotype 原型只能在 outer-train discovery 患者上拟合；"
            "calibration 和 holdout 只能做冻结归属。",
            size=9.5,
        )
        add_text(fig, 0.06, 0.395, "患者个性化规则", size=13, weight="bold", color="#183A61")
        add_text(
            fig,
            0.08,
            0.36,
            "对患者前半段中每个 phenotype，分别统计 SingLEM、CBraMod 和 TCN-GAT 的：\n"
            "• TP 数量和 sensitivity；\n"
            "• FP 数量和 FA/24h；\n"
            "• detection delay。\n\n"
            "保留 sensitivity 接近最高模型的候选，再选择 FA/24h 最低的模型。"
            "整个模型选择规则在前半段结束时冻结。",
            size=9.5,
        )
        boxed_text(
            fig,
            0.06,
            0.22,
            0.88,
            0.18,
            "样本不足时的 fallback",
            "如果某患者前半段没有 seizure，或某 phenotype 事件太少，\n"
            "不允许硬拟合个性化规则。\n"
            "该 phenotype 使用预先冻结的全局规则，并单独报告 fallback 比例。\n"
            "推荐先做 post-hoc routing，复用现有缓存概率；\n"
            "确认有效后再考虑 pre-model routing。",
            face="#F4F4F4",
            edge="#888888",
            title_color="#444444",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4
        fig = page_base("3. 对照实验和评价指标", 4, "必须证明提升来自个性化，而不是简单放宽阈值")
        ax = fig.add_axes([0.06, 0.53, 0.88, 0.28])
        ax.axis("off")
        table_data = [
            ["方法", "使用前半段标签", "作用"],
            ["全局最佳单模型", "否", "患者无关 baseline"],
            ["固定等权 ensemble", "否", "检查简单融合"],
            ["当前通用 router", "否", "检查一般 router 是否足够"],
            ["患者自适应模型选择", "是", "不分 phenotype 的个性化 baseline"],
            ["患者 phenotype-router", "是", "主要实验方法"],
        ]
        table = ax.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.28, 0.22, 0.5])
        table.scale(1, 1.65)
        style_table(table, fontsize=8.6)
        add_text(fig, 0.06, 0.475, "主要指标", size=13, weight="bold", color="#183A61")
        add_text(
            fig,
            0.08,
            0.44,
            "• event sensitivity\n"
            "• FA/24h\n"
            "• detection delay\n"
            "• 患者级 95% bootstrap CI\n"
            "• 每位患者的 waterfall\n"
            "• calibration 阶段可适应比例和 fallback 比例",
            size=9.5,
        )
        boxed_text(
            fig,
            0.50,
            0.46,
            0.44,
            0.25,
            "Primary gate",
            "后半段 FA/24h 至少下降 25%，且 sensitivity 相对全局最佳单模型最多下降 3 个百分点。\n\n"
            "门槛必须在读取 holdout 结果前冻结。",
            face="#EEF7EE",
            edge="#6BA56B",
            title_color="#2F6B2F",
        )
        add_text(fig, 0.06, 0.225, "统计分析", size=13, weight="bold", color="#183A61")
        add_text(
            fig,
            0.08,
            0.19,
            "以患者为统计单位，做 paired comparison。报告每位患者的个性化方法减去 baseline 的变化，"
            "并使用患者级 bootstrap 或配对置换检验。seed 作为重复因素，不把多个窗口当作独立患者。",
            size=9.5,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 5
        fig = page_base("4. 防止过拟合和结果解释", 5, "有标签校准可以做，但必须承认它改变了研究问题")
        add_text(fig, 0.06, 0.875, "必须锁定的审计项目", size=13, weight="bold", color="#183A61")
        checks = [
            "calibration 与 holdout 不共享同一 recording 或重叠窗口",
            "phenotype 原型和 scaler 不读取 holdout 标签",
            "holdout 不参与模型选择、阈值选择和 phenotype 数量选择",
            "没有 seizure 的患者不被删除，而是走 fallback 并计数",
            "保留全部患者，即使个性化规则没有足够样本",
            "最终报告同时展示 calibration 表现和 holdout 表现",
            "所有配置、split manifest、feature hash 和 routing map 保存",
        ]
        y = 0.83
        for check in checks:
            add_text(fig, 0.085, y, "OK", size=9, weight="bold", color="#2F7D54")
            add_text(fig, 0.13, y, check, size=9.5)
            y -= 0.055
        boxed_text(
            fig,
            0.06,
            0.43,
            0.88,
            0.17,
            "如何解释阳性结果",
            "如果后半段 FA/24h 降低且 sensitivity 不下降，\n"
            "可以说早期有标签校准支持患者内个性化路由。\n"
            "不能说模型对所有患者都更好，\n"
            "也不能把它当作没有适应的外部泛化。",
            face="#EEF7EE",
            edge="#6BA56B",
            title_color="#2F6B2F",
        )
        boxed_text(
            fig,
            0.06,
            0.22,
            0.88,
            0.18,
            "如何解释阴性结果",
            "如果 calibration 表现很好但后半段失败，\n"
            "说明个性化规则可能过拟合早期患者状态或 phenotype 不稳定。\n"
            "如果多数患者走 fallback，结论应是当前数据不足以支持个性化路由。\n"
            "这是一个 supervised personalization proof-of-concept，\n"
            "不是无标签、即时部署的最终系统。",
            face="#FFF4F4",
            edge="#C46B6B",
            title_color="#8A3030",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 6
        fig = page_base("5. 执行时间和产物", 6, "优先复用现有标准 GroupKFold 概率和特征缓存")
        ax = fig.add_axes([0.06, 0.52, 0.88, 0.32])
        ax.axis("off")
        schedule = [
            ["阶段", "工作", "预计时间"],
            ["1", "冻结 namespace、时间 split、审计规则、25-update smoke", "1-2 小时"],
            ["2", "事件特征和 phenotype assignment", "2-4 小时"],
            ["3", "五种方法的 calibration/holdout 推理", "2-4 小时"],
            ["4", "患者 bootstrap、waterfall、配对统计", "1-3 小时"],
            ["5", "图表、manifest、结果报告", "2-4 小时"],
        ]
        table = ax.table(cellText=schedule, loc="center", cellLoc="center", colWidths=[0.12, 0.58, 0.3])
        table.scale(1, 1.7)
        style_table(table, fontsize=8.4)
        add_text(fig, 0.06, 0.445, "总时间估计", size=13, weight="bold", color="#183A61")
        add_text(
            fig,
            0.08,
            0.41,
            "• 复用现有概率和缓存：约 8-15 小时，通常一天内完成。\n"
            "• 如果需要重新生成全量推理：额外约 12-24 小时。\n"
            "• 不建议第一步做患者级 backbone fine-tuning；那会增加到数天并提高过拟合风险。",
            size=9.8,
        )
        add_text(fig, 0.06, 0.255, "最终产物", size=13, weight="bold", color="#183A61")
        add_text(
            fig,
            0.08,
            0.22,
            "1. calibration/holdout manifest 和 hash\n"
            "2. 每位患者的 phenotype-to-model routing map\n"
            "3. 五种方法的 holdout metrics\n"
            "4. 患者 bootstrap CI 和 waterfall\n"
            "5. adaptation coverage/fallback table\n"
            "6. H6 结果报告和限制说明",
            size=9.5,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 7
        fig = page_base("6. 常见问题和最终决策规则", 7, "这项实验完成后应该声称什么")
        qa = [
            (
                "Q1. 这是重新训练模型吗？",
                "不是。主实验冻结三个 backbone 和 CHB 阈值，只用前半段标签学习患者专属的模型选择关系。",
            ),
            (
                "Q2. 后半段能不能调规则？",
                "不能。后半段是一次性 chronological holdout，只能用于最终评估。",
            ),
            (
                "Q3. 如果患者前半段没有 seizure 怎么办？",
                "不删除该患者。使用全局 fallback，并报告该患者不能完成 supervised personalization。",
            ),
            (
                "Q4. 阳性结果能证明什么？",
                "能证明早期有标签校准可能改善同一患者后期检测；不能证明对新患者也一定有效。",
            ),
            (
                "Q5. 如果 router 只提高 sensitivity 但增加 FA 呢？",
                "按预设 gate 判为不支持。那只是 operating-point trade-off，不是有效的假报警抑制。",
            ),
            (
                "Q6. 它和当前 H4 的关系是什么？",
                "当前 H4 是跨患者的通用 router；H6 是使用患者早期标签的个性化 router，必须使用新的时间留出。",
            ),
        ]
        y = 0.87
        for question, answer in qa:
            add_text(fig, 0.07, y, question, size=10.5, weight="bold", color="#183A61")
            add_text(fig, 0.09, y - 0.036, answer, size=9.0)
            y -= 0.115
        boxed_text(
            fig,
            0.06,
            0.18,
            0.88,
            0.13,
            "最终定位",
            "H6 是一个新的 within-patient personalization extension。\n"
            "它不覆盖、不修改当前标准 v3、H1-H5 或 TUSZ 外部验证结果。",
            face="#F4F7FB",
            edge="#7A9CC6",
            title_color="#183A61",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    make_pdf()
    print(OUT)
