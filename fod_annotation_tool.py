#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FOD (Foreign Object Debris) 图像标注工具
=========================================
功能：
  1. 浏览本地图片，左右按钮快速切换
  2. 支持放大/缩小，便于标注细小物体
  3. 12个FOD类别按钮，点击选择类别后画框标注
  4. 标注框(BBox)微调：四个参数独立调节
  5. "下一个"存储标注，"确定"保存到同名txt（JSON结构体，{}分级）
  6. 自动加载已有标注文件
  7. 切换图片自动保存

依赖: tkinter (标准库), Pillow
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

# tifffile 用于读取 PIL 无法处理的 TIFF 格式
try:
    import tifffile as _tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

# ──────────────────────────────────────────────────────────────
# 配置: 12个FOD类别（0-11编号对应存储）
# ──────────────────────────────────────────────────────────────
DEFAULT_CATEGORIES = [
    "金属", "橡胶", "塑料", "木头",
    "纸板", "织物", "动物", "玻璃",
    "石头", "泡沫", "油渍", "水",
]

# 类别 ↔ 编号映射（txt中存储0-11的整数编号）
CATEGORY_TO_ID = {name: i for i, name in enumerate(DEFAULT_CATEGORIES)}
ID_TO_CATEGORY = {i: name for i, name in enumerate(DEFAULT_CATEGORIES)}

# 支持的图片扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}


@dataclass
class Annotation:
    """单个标注数据结构 — 内部使用像素坐标，序列化为归一化比例(0~1)"""
    category: str          # 类别名称（显示用）
    x1: int                # 左上X (像素)
    y1: int                # 左上Y (像素)
    x2: int                # 右下X (像素)
    y2: int                # 右下Y (像素)

    def to_dict(self, img_w: int, img_h: int) -> Dict[str, Any]:
        """序列化为结构化字典，bbox用归一化比例存储（0~1，缩放不变）"""
        return {
            "id": CATEGORY_TO_ID.get(self.category, -1),
            "name": self.category,
            "bbox": {
                "x1": round(self.x1 / img_w, 6),
                "y1": round(self.y1 / img_h, 6),
                "x2": round(self.x2 / img_w, 6),
                "y2": round(self.y2 / img_h, 6),
            }
        }

    @staticmethod
    def from_dict(d: Dict[str, Any], img_w: int = 0, img_h: int = 0) -> Optional['Annotation']:
        """从结构化字典反序列化，自动兼容旧格式（像素值>1）和新格式（归一化0~1）"""
        try:
            class_id = int(d["id"])
            bbox = d["bbox"]
            x1, y1 = float(bbox["x1"]), float(bbox["y1"])
            x2, y2 = float(bbox["x2"]), float(bbox["y2"])
            # 检测格式：若任一值 > 1 则为旧格式像素值，否则为新格式比例
            if max(x1, y1, x2, y2) > 1.0 and img_w > 0 and img_h > 0:
                # 旧格式：绝对像素 → 直接取整
                x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            elif img_w > 0 and img_h > 0:
                # 新格式：归一化比例 → 还原为像素
                x1 = int(round(x1 * img_w))
                y1 = int(round(y1 * img_h))
                x2 = int(round(x2 * img_w))
                y2 = int(round(y2 * img_h))
            else:
                # 无图像尺寸时保持原值
                x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            category = ID_TO_CATEGORY.get(class_id, d.get("name", f"未知{class_id}"))
            return Annotation(category=category, x1=x1, y1=y1, x2=x2, y2=y2)
        except (KeyError, ValueError, TypeError):
            return None


class FODAnnotationTool:
    """FOD标注工具主界面"""

    # ── 初始化 ──────────────────────────────────────────────
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FOD 图像标注工具")
        self.root.geometry("2080x1300")
        self.root.minsize(1690, 1040)

        # ── 状态变量 ──
        self.image_folder: str = ""
        self.image_list: List[str] = []
        self.current_image_idx: int = -1
        self.pil_image: Optional[Image.Image] = None   # 显示用图像（可能已下采样）
        self.original_image_size: Tuple[int, int] = (0, 0)  # 原始图像尺寸（标注保存依据）
        self.display_scale: float = 1.0               # 显示/原始比例 (1.0 或 1/3)
        self.downsample_enabled: bool = False          # 是否启用3倍下采样
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.zoom_scale: float = 1.0
        self.canvas_img_id: Optional[int] = None       # Canvas上图像对象的ID

        self.annotations: List[Annotation] = []        # 当前图像的所有标注
        self.current_category: Optional[str] = None
        self.current_bbox: Optional[List[int]] = None  # [x1, y1, x2, y2] 图像坐标
        self.preview_rect_id: Optional[int] = None     # 红色预览框的Canvas ID
        self.saved_rect_ids: List[int] = []            # 已保存标注的Canvas框ID列表

        self.drawing: bool = False
        self.drag_start: Tuple[float, float] = (0, 0)

        self.save_directory: str = ""                  # 自定义保存目录（空=图片同目录）

        # 有效工作区域
        self.work_region: Optional[List[int]] = None   # [x1, y1, x2, y2] 图像坐标
        self.drawing_work_region: bool = False         # 正在绘制工作区域
        self.work_region_rect_id: Optional[int] = None  # 工作区域绿色框 Canvas ID
        self.work_region_dim_ids: List[int] = []        # 工作区域角标 Canvas ID
        self.work_region_adjust_mode: bool = False     # 微调面板控制工作区域模式

        # 配准变换 (双相机投影)
        self.reg_points_src: List[Tuple[int, int]] = []   # 源图配准点(图像坐标)
        self.reg_points_dst: List[Tuple[int, int]] = []   # 目标图配准点(图像坐标)
        self.registration_active: bool = False            # 配准点采集模式
        self.registration_stage: str = ""                 # "src" / "dst"
        self.reg_marker_ids: List[int] = []               # 配准点标记 Canvas ID
        self.homography_matrix: Optional[np.ndarray] = None  # 3x3 单应性矩阵
        self.reg_source_thumbnail: Optional[ImageTk.PhotoImage] = None  # 源图缩略图(含配准点)
        self.reg_thumb_id: Optional[int] = None            # 缩略图 Canvas ID
        self.source_annotation_folder: str = ""            # 源标注文件夹（批量投影）
        self.auto_project_enabled: bool = False            # 自动投影开关

        # 长按连续调整
        self._adjust_repeat_id: Optional[str] = None       # after ID for repeat

        # ── 构建界面 ──
        self._setup_menu()
        self._setup_ui()
        self._bind_events()

        # 状态栏
        self._update_status("就绪 — 请通过「文件 → 打开文件夹」选择图片目录")

    # ══════════════════════════════════════════════════════════
    # 菜单
    # ══════════════════════════════════════════════════════════
    def _setup_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开文件夹...", command=self.open_folder, accelerator="Ctrl+O")
        file_menu.add_command(label="选择保存目录...", command=self.select_save_dir)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self._on_close)
        menubar.add_cascade(label="文件", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于", command=self._show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)
        self.root.config(menu=menubar)
        self.root.bind('<Control-o>', lambda e: self.open_folder())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════════
    # 主界面布局
    # ══════════════════════════════════════════════════════════
    def _setup_ui(self):
        # 主容器：左右分栏
        main_pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pw.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        # ── 左侧: 图像显示区 ──
        left_frame = ttk.Frame(main_pw)
        main_pw.add(left_frame, weight=3)

        # Canvas + 滚动条
        canvas_container = ttk.Frame(left_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_container, bg='#2d2d2d',
                                cursor='crosshair', highlightthickness=0)
        self.canvas_hbar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL,
                                         command=self.canvas.xview)
        self.canvas_vbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL,
                                         command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.canvas_hbar.set,
                              yscrollcommand=self.canvas_vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.canvas_hbar.grid(row=1, column=0, sticky='ew')
        self.canvas_vbar.grid(row=0, column=1, sticky='ns')

        # 缩放控制栏
        ctrl_bar = ttk.Frame(left_frame)
        ctrl_bar.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(ctrl_bar, text="放大 +", command=self.zoom_in, width=7).pack(side=tk.LEFT, padx=1)
        ttk.Button(ctrl_bar, text="缩小 -", command=self.zoom_out, width=7).pack(side=tk.LEFT, padx=1)
        ttk.Button(ctrl_bar, text="适应窗口", command=self.fit_to_window, width=8).pack(side=tk.LEFT, padx=1)
        ttk.Button(ctrl_bar, text="原始大小", command=self.zoom_100, width=8).pack(side=tk.LEFT, padx=1)
        self.zoom_label = ttk.Label(ctrl_bar, text="100%", width=8, anchor='center')
        self.zoom_label.pack(side=tk.LEFT, padx=10)

        # 下采样切换
        self.downsample_var = tk.BooleanVar(value=False)
        self.downsample_cb = ttk.Checkbutton(
            ctrl_bar, text="3倍下采样", variable=self.downsample_var,
            command=self._toggle_downsample)
        self.downsample_cb.pack(side=tk.LEFT, padx=10)

        # 导航栏
        nav_bar = ttk.Frame(left_frame)
        nav_bar.pack(fill=tk.X, padx=4, pady=4)
        self.btn_prev = ttk.Button(nav_bar, text="◀ 上一张", command=self.prev_image)
        self.btn_prev.pack(side=tk.LEFT, padx=2)

        # 图片快速跳转下拉框 + 位置标签
        jump_frame = ttk.Frame(nav_bar)
        jump_frame.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        ttk.Label(jump_frame, text="跳转:").pack(side=tk.LEFT, padx=(0, 3))
        self._combo_var = tk.StringVar()
        self.image_combo = ttk.Combobox(jump_frame, textvariable=self._combo_var,
                                        state='readonly', width=30)
        self.image_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.image_combo.bind('<<ComboboxSelected>>', self._on_combo_select)
        self.pos_label = ttk.Label(jump_frame, text="", width=12, anchor='center')
        self.pos_label.pack(side=tk.RIGHT, padx=(6, 0))

        self.btn_next = ttk.Button(nav_bar, text="下一张 ▶", command=self.next_image)
        self.btn_next.pack(side=tk.RIGHT, padx=2)

        # ── 右侧: 控制面板 ──
        right_frame = ttk.Frame(main_pw, width=380)
        main_pw.add(right_frame, weight=1)

        # 类别选择区域（可滚动）
        cat_lf = ttk.LabelFrame(right_frame, text="标注类别 (点击选择)")
        cat_lf.pack(fill=tk.BOTH, expand=True, padx=3, pady=2)

        # 用 Canvas 实现滚动 (12个类别，3行×4列)
        cat_canvas = tk.Canvas(cat_lf, height=230, highlightthickness=0)
        cat_scroll = ttk.Scrollbar(cat_lf, orient=tk.VERTICAL, command=cat_canvas.yview)
        self.cat_inner = ttk.Frame(cat_canvas)
        self.cat_inner.bind('<Configure>',
                            lambda e: cat_canvas.configure(scrollregion=cat_canvas.bbox('all')))
        self._cat_win = cat_canvas.create_window((0, 0), window=self.cat_inner, anchor='nw')

        # 让内部frame宽度跟随canvas
        def _on_cat_canvas_conf(e):
            cat_canvas.itemconfig(self._cat_win, width=e.width)
        cat_canvas.bind('<Configure>', _on_cat_canvas_conf)

        cat_canvas.configure(yscrollcommand=cat_scroll.set)
        cat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cat_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 绑定鼠标滚轮
        def _cat_mousewheel(e):
            cat_canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')
        cat_canvas.bind('<Enter>',
                        lambda e: cat_canvas.bind_all('<MouseWheel>', _cat_mousewheel))
        cat_canvas.bind('<Leave>',
                        lambda e: cat_canvas.unbind_all('<MouseWheel>'))

        # 生成12个类别按钮 (4列布局)
        self.cat_buttons: List[ttk.Button] = []
        self._cat_btn_ref: dict = {}  # category -> button
        for i, cat_name in enumerate(DEFAULT_CATEGORIES):
            row, col = divmod(i, 4)
            btn = ttk.Button(self.cat_inner, text=cat_name, width=8,
                             command=lambda c=cat_name: self.select_category(c))
            btn.grid(row=row, column=col, padx=3, pady=3, sticky='ew')
            self.cat_buttons.append(btn)
            self._cat_btn_ref[cat_name] = btn
        # 均匀列宽
        for c in range(4):
            self.cat_inner.grid_columnconfigure(c, weight=1)

        # 当前选中类别提示
        self.cat_hint = ttk.Label(right_frame, text="当前类别: 未选择",
                                  foreground='#888', font=('Microsoft YaHei UI', 22, 'bold'))
        self.cat_hint.pack(fill=tk.X, padx=5, pady=(2, 0))

        # ── 有效工作区域 ──
        wr_lf = ttk.LabelFrame(right_frame, text="有效工作区域")
        wr_lf.pack(fill=tk.X, padx=3, pady=4)

        wr_btn_row = ttk.Frame(wr_lf)
        wr_btn_row.pack(fill=tk.X, padx=4, pady=3)
        self.wr_toggle_btn = ttk.Button(wr_btn_row, text="设置工作区域",
                                         command=self._toggle_work_region_mode)
        self.wr_toggle_btn.pack(side=tk.LEFT, padx=2)
        self.wr_clear_btn = ttk.Button(wr_btn_row, text="清除",
                                        command=self._clear_work_region)
        self.wr_clear_btn.pack(side=tk.LEFT, padx=2)
        self.wr_status_label = ttk.Label(wr_btn_row, text="未设置", foreground='gray')
        self.wr_status_label.pack(side=tk.LEFT, padx=6)

        wr_adj_row = ttk.Frame(wr_lf)
        wr_adj_row.pack(fill=tk.X, padx=4, pady=(0, 3))
        self.wr_adjust_var = tk.BooleanVar(value=False)
        self.wr_adjust_cb = ttk.Checkbutton(
            wr_adj_row, text="微调模式", variable=self.wr_adjust_var,
            command=self._toggle_work_region_adjust)
        self.wr_adjust_cb.pack(side=tk.LEFT, padx=2)
        ttk.Label(wr_adj_row, text="(勾选后左侧微调面板控制工作区域)",
                  foreground='gray').pack(side=tk.LEFT, padx=4)

        # ── 配准变换 (双相机投影) ──
        reg_lf = ttk.LabelFrame(right_frame, text="配准变换 (双相机投影)")
        reg_lf.pack(fill=tk.X, padx=3, pady=4)

        # 配准点采集行
        reg_pt_row = ttk.Frame(reg_lf)
        reg_pt_row.pack(fill=tk.X, padx=4, pady=3)
        self.reg_src_btn = ttk.Button(reg_pt_row, text="采源图点",
                                       command=lambda: self._start_registration("src"))
        self.reg_src_btn.pack(side=tk.LEFT, padx=1)
        self.reg_dst_btn = ttk.Button(reg_pt_row, text="采目标图点",
                                       command=lambda: self._start_registration("dst"))
        self.reg_dst_btn.pack(side=tk.LEFT, padx=1)
        self.reg_undo_btn = ttk.Button(reg_pt_row, text="撤销上一点",
                                        command=self._undo_last_registration_point)
        self.reg_undo_btn.pack(side=tk.LEFT, padx=1)
        self.reg_clear_btn = ttk.Button(reg_pt_row, text="清除全部点",
                                         command=self._clear_registration)
        self.reg_clear_btn.pack(side=tk.LEFT, padx=1)
        self.reg_status_label = ttk.Label(reg_pt_row, text="源:0 目标:0", foreground='gray')
        self.reg_status_label.pack(side=tk.LEFT, padx=6)

        # 矩阵操作行
        reg_mat_row = ttk.Frame(reg_lf)
        reg_mat_row.pack(fill=tk.X, padx=4, pady=(0, 3))
        self.reg_compute_btn = ttk.Button(reg_mat_row, text="计算并保存矩阵",
                                           command=self._compute_and_save_homography)
        self.reg_compute_btn.pack(side=tk.LEFT, padx=1)
        self.reg_load_btn = ttk.Button(reg_mat_row, text="加载矩阵",
                                        command=self._load_homography_from_file)
        self.reg_load_btn.pack(side=tk.LEFT, padx=1)
        self.reg_matrix_label = ttk.Label(reg_mat_row, text="矩阵: 未设置", foreground='gray')
        self.reg_matrix_label.pack(side=tk.LEFT, padx=4)

        # 投影操作行
        reg_proj_row = ttk.Frame(reg_lf)
        reg_proj_row.pack(fill=tk.X, padx=4, pady=(0, 3))
        self.reg_project_btn = ttk.Button(reg_proj_row, text="从源图投影标注",
                                           command=self._project_annotations_from_source)
        self.reg_project_btn.pack(side=tk.LEFT, padx=1)

        # 批量投影行
        reg_batch_row = ttk.Frame(reg_lf)
        reg_batch_row.pack(fill=tk.X, padx=4, pady=(0, 3))
        self.reg_folder_btn = ttk.Button(reg_batch_row, text="源标注文件夹",
                                          command=self._select_source_annotation_folder)
        self.reg_folder_btn.pack(side=tk.LEFT, padx=1)
        self.auto_project_var = tk.BooleanVar(value=False)
        self.auto_project_cb = ttk.Checkbutton(
            reg_batch_row, text="自动投影", variable=self.auto_project_var,
            command=self._toggle_auto_project)
        self.auto_project_cb.pack(side=tk.LEFT, padx=4)

        # 文件夹状态行
        reg_status_row = ttk.Frame(reg_lf)
        reg_status_row.pack(fill=tk.X, padx=4, pady=(0, 3))
        self.reg_folder_status = ttk.Label(reg_status_row, text="源文件夹: 未选择",
                                           foreground='gray')
        self.reg_folder_status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ── 标注框微调 ──
        adj_lf = ttk.LabelFrame(right_frame, text="标注框微调 (图像坐标)")
        adj_lf.pack(fill=tk.X, padx=3, pady=4)

        self.adjust_vars: dict = {}
        self.adjust_entries: dict = {}
        params = [
            ('x1', '左'), ('y1', '上'),
            ('x2', '右'), ('y2', '下'),
        ]
        for key, label in params:
            row_f = ttk.Frame(adj_lf)
            row_f.pack(fill=tk.X, padx=4, pady=2)
            ttk.Label(row_f, text=f"{label}:", width=7).pack(side=tk.LEFT)

            var = tk.IntVar(value=0)
            self.adjust_vars[key] = var

            btn_minus = ttk.Button(row_f, text="◀", width=3,
                                    command=lambda k=key: self.adjust_box(k, -1))
            btn_minus.bind('<ButtonPress-1>', lambda e, k=key: self._start_repeat_timer(k, -1))
            btn_minus.bind('<ButtonRelease-1>', lambda e: self._stop_adjust_repeat())
            btn_minus.bind('<Leave>', lambda e: self._stop_adjust_repeat())
            btn_minus.pack(side=tk.LEFT, padx=1)
            entry = ttk.Entry(row_f, textvariable=var, width=7, justify='center')
            entry.pack(side=tk.LEFT, padx=2)
            entry.bind('<Return>', lambda e, k=key: self._on_entry_commit(k))
            entry.bind('<FocusOut>', lambda e, k=key: self._on_entry_commit(k))
            self.adjust_entries[key] = entry
            btn_plus = ttk.Button(row_f, text="▶", width=3,
                                   command=lambda k=key: self.adjust_box(k, 1))
            btn_plus.bind('<ButtonPress-1>', lambda e, k=key: self._start_repeat_timer(k, 1))
            btn_plus.bind('<ButtonRelease-1>', lambda e: self._stop_adjust_repeat())
            btn_plus.bind('<Leave>', lambda e: self._stop_adjust_repeat())
            btn_plus.pack(side=tk.LEFT, padx=1)

        # 步长选择
        step_f = ttk.Frame(adj_lf)
        step_f.pack(fill=tk.X, padx=4, pady=3)
        ttk.Label(step_f, text="微调步长:").pack(side=tk.LEFT)
        self.step_var = tk.IntVar(value=1)
        ttk.Spinbox(step_f, from_=1, to=100, textvariable=self.step_var,
                    width=6).pack(side=tk.LEFT, padx=6)

        # 标注框尺寸信息
        size_f = ttk.Frame(adj_lf)
        size_f.pack(fill=tk.X, padx=4, pady=(2, 0))
        self.bbox_size_label = ttk.Label(size_f, text="宽: —  高: —  面积: —",
                                         foreground='#555', anchor='center')
        self.bbox_size_label.pack(fill=tk.X)

        # ── 操作按钮 ──
        btn_f = ttk.Frame(right_frame)
        btn_f.pack(fill=tk.X, padx=3, pady=4)
        ttk.Button(btn_f, text="下一个", command=self.next_annotation).pack(
            side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(btn_f, text="确定 ✓", command=self.confirm_image).pack(
            side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(btn_f, text="删除选中", command=self.delete_annotation).pack(
            side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        # 清空当前标注(撤销当前正在画的框)
        ttk.Button(btn_f, text="撤销框", command=self.clear_current_bbox).pack(
            side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # ── 保存目录 ──
        save_f = ttk.Frame(right_frame)
        save_f.pack(fill=tk.X, padx=3, pady=2)
        ttk.Label(save_f, text="保存至:").pack(side=tk.LEFT)
        self.save_dir_label = ttk.Label(save_f, text="(图片同目录)", foreground='gray')
        self.save_dir_label.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        ttk.Button(save_f, text="选择...", command=self.select_save_dir).pack(side=tk.RIGHT)

        # ── 底部: 标注结果列表 ──
        bottom_frame = ttk.LabelFrame(self.root, text="标注结果")
        bottom_frame.pack(fill=tk.BOTH, expand=False, padx=3, pady=2, side=tk.BOTTOM)

        columns = ('序号', 'ID', '类别', 'x1', 'y1', 'x2', 'y2')
        self.tree = ttk.Treeview(bottom_frame, columns=columns, show='headings',
                                 height=6, selectmode='browse')
        col_widths = [50, 40, 80, 70, 70, 70, 70]
        for col, w in zip(columns, col_widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor='center')
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_sb = ttk.Scrollbar(bottom_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_sb.set)
        tree_sb.pack(side=tk.RIGHT, fill=tk.Y)

        # 双击 Treeview 行 → 选中并加载到画布进行修改
        self.tree.bind('<Double-1>', self._on_tree_double_click)

        # ── 状态栏 ──
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W, padding=(6, 6),
                               font=('Microsoft YaHei UI', 19))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ══════════════════════════════════════════════════════════
    # 事件绑定
    # ══════════════════════════════════════════════════════════
    def _bind_events(self):
        # 鼠标画框
        self.canvas.bind('<ButtonPress-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)

        # 滚轮缩放
        self.canvas.bind('<Control-MouseWheel>', self._on_ctrl_wheel)
        # Windows 上 MouseWheel, Linux 上 Button-4/5
        self.canvas.bind('<MouseWheel>', self._on_mouse_wheel)
        self.canvas.bind('<Button-4>', self._on_mouse_wheel)
        self.canvas.bind('<Button-5>', self._on_mouse_wheel)

        # 键盘快捷键
        self.root.bind('<Control-plus>', lambda e: self.zoom_in())
        self.root.bind('<Control-minus>', lambda e: self.zoom_out())
        self.root.bind('<Control-0>', lambda e: self.zoom_100())
        self.root.bind('<Left>', lambda e: self._on_arrow('left'))
        self.root.bind('<Right>', lambda e: self._on_arrow('right'))
        self.root.bind('<Up>', lambda e: self._on_arrow('up'))
        self.root.bind('<Down>', lambda e: self._on_arrow('down'))
        self.root.bind('<Escape>', lambda e: self._on_escape())

    # ══════════════════════════════════════════════════════════
    # 文件 / 图片操作
    # ══════════════════════════════════════════════════════════
    def open_folder(self):
        """打开图片文件夹"""
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if not folder:
            return
        self.image_folder = folder
        # 扫描图片
        self.image_list = []
        for fname in sorted(os.listdir(folder)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                self.image_list.append(os.path.join(folder, fname))
        if not self.image_list:
            messagebox.showinfo("提示", "所选文件夹中没有找到图片文件。")
            return
        self.current_image_idx = 0
        self._load_and_display()

    @staticmethod
    def _normalize_numpy(arr: np.ndarray) -> np.ndarray:
        """将任意dtype的numpy数组归一化到8-bit [0, 255]"""
        arr = arr.astype(np.float64, copy=True)
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if vmax == vmin:
            return np.zeros(arr.shape, dtype=np.uint8)
        normalized = (arr - vmin) / (vmax - vmin) * 255.0
        return normalized.clip(0, 255).astype(np.uint8)

    @staticmethod
    def _numpy_to_pil(arr: np.ndarray) -> Image.Image:
        """将numpy数组(2D/3D)转换为PIL Image，自动归一化"""
        if arr.ndim == 2:
            # 单通道灰度
            norm = FODAnnotationTool._normalize_numpy(arr)
            return Image.fromarray(norm, mode='L')
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
                # (C, H, W) channel-first 格式
                if arr.shape[0] == 1:
                    return FODAnnotationTool._numpy_to_pil(arr[0])
                elif arr.shape[0] == 3:
                    r = FODAnnotationTool._normalize_numpy(arr[0])
                    g = FODAnnotationTool._normalize_numpy(arr[1])
                    b = FODAnnotationTool._normalize_numpy(arr[2])
                    return Image.fromarray(np.stack([r, g, b], axis=-1), mode='RGB')
                else:
                    # >3 channels: take first 3
                    r = FODAnnotationTool._normalize_numpy(arr[0])
                    g = FODAnnotationTool._normalize_numpy(arr[1])
                    b = FODAnnotationTool._normalize_numpy(arr[2])
                    return Image.fromarray(np.stack([r, g, b], axis=-1), mode='RGB')
            elif arr.shape[2] in (1, 3, 4):
                # (H, W, C) channel-last 格式
                if arr.shape[2] == 1:
                    return FODAnnotationTool._numpy_to_pil(arr[:, :, 0])
                elif arr.shape[2] >= 3:
                    r = FODAnnotationTool._normalize_numpy(arr[:, :, 0])
                    g = FODAnnotationTool._normalize_numpy(arr[:, :, 1])
                    b = FODAnnotationTool._normalize_numpy(arr[:, :, 2])
                    return Image.fromarray(np.stack([r, g, b], axis=-1), mode='RGB')
        # 兜底
        arr_2d = arr.reshape(arr.shape[0], -1)
        return FODAnnotationTool._numpy_to_pil(arr_2d)

    def _load_image(self, img_path: str) -> Image.Image:
        """加载图片：PIL优先，TIFF兜底使用tifffile → numpy → PIL"""
        ext = os.path.splitext(img_path)[1].lower()
        is_tiff = ext in ('.tiff', '.tif')

        # ── 先尝试 PIL ──
        pil_error = None
        try:
            img = Image.open(img_path)
            img.load()  # 触发实际解码
            # 对TIFF做快速检测：如果能正常获取像素信息则用PIL
            if is_tiff:
                try:
                    extrema = img.getextrema()
                    if extrema is None:
                        raise ValueError("PIL getextrema 返回 None")
                except Exception:
                    pil_error = Exception("PIL无法解析该TIFF的像素数据")
            if pil_error is None:
                return self._normalize_image(img)
        except Exception as e:
            pil_error = e

        # ── PIL 失败时，对 TIFF 使用 tifffile ──
        if is_tiff and HAS_TIFFFILE:
            try:
                arr = _tifffile.imread(img_path)
                arr = np.asarray(arr)
                # 处理多页TIFF
                if arr.ndim == 4 and arr.shape[0] > 1:
                    arr = arr[0]  # 取第一页
                # 若有batch维 (1, H, W) 或 (B, C, H, W)
                while arr.ndim > 3:
                    arr = arr[0]
                img = self._numpy_to_pil(arr)
                return self._normalize_image(img)
            except Exception as e2:
                raise Exception(
                    f"PIL 和 tifffile 均无法读取该图片:\n"
                    f"  PIL: {pil_error}\n"
                    f"  tifffile: {e2}"
                )

        # ── 彻底失败 ──
        if pil_error:
            raise pil_error
        raise Exception("无法读取该图片格式")

    def _normalize_image(self, img: Image.Image) -> Image.Image:
        """将任意bit深度/颜色模式的图像归一化为8-bit RGB，适配TIFF等高位深图片"""
        mode = img.mode

        # ── 处理多页TIFF：只取第一页 ──
        try:
            img.seek(0)
        except Exception:
            pass

        # ── 模式转换：P(调色板) / 1(二值) / CMYK ──
        if mode == '1':
            img = img.convert('L')
            mode = 'L'
        elif mode == 'P':
            img = img.convert('RGBA')
            mode = 'RGBA'
        elif mode == 'CMYK':
            img = img.convert('RGB')
            mode = 'RGB'

        # ── 高位深灰度归一化到8-bit ──
        high_depth_modes = {'I', 'F', 'I;16', 'I;16B', 'I;16L', 'I;32', 'I;32L'}
        if mode in high_depth_modes:
            # 统一转换到 I (32-bit) 方便处理
            if mode != 'I':
                try:
                    img = img.convert('I')
                except Exception:
                    pass
            # 获取像素值范围并拉伸到 0~255
            extrema = img.getextrema()
            vmin, vmax = extrema[0], extrema[1]
            if vmax > vmin:
                scale = 255.0 / (vmax - vmin)
                img = img.point(lambda i: int(round((i - vmin) * scale)))
            else:
                img = img.point(lambda i: 0)
            img = img.convert('L')
            mode = 'L'

        # ── 处理透明度通道 ──
        if mode == 'RGBA':
            bg = Image.new('RGB', img.size, (128, 128, 128))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif mode == 'LA':
            bg = Image.new('L', img.size, 255)
            bg.paste(img, mask=img.split()[1])
            img = bg.convert('RGB')
        elif mode == 'L':
            img = img.convert('RGB')

        # ── 兜底：确保最终为 RGB ──
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return img

    def _load_and_display(self):
        """加载当前索引的图片并显示，自动搜索同名txt"""
        if self.current_image_idx < 0 or self.current_image_idx >= len(self.image_list):
            return

        img_path = self.image_list[self.current_image_idx]
        try:
            raw_img = self._load_image(img_path)
        except Exception as e:
            messagebox.showerror("错误", f"无法打开图片:\n{img_path}\n\n{e}")
            return

        # 记录原始尺寸（标注保存/加载依据）
        self.original_image_size = raw_img.size

        # 应用下采样（如启用）
        if self.downsample_enabled:
            self.pil_image = self._downsample_3x(raw_img)
            self.display_scale = 1.0 / 3.0
        else:
            self.pil_image = raw_img
            self.display_scale = 1.0

        # 保持当前缩放比例（首次加载时用fit_to_window初始化）
        if self.zoom_scale <= 0.05:
            self.fit_to_window()

        # 清空标注
        self.annotations.clear()
        self.work_region = None
        self.wr_status_label.config(text="未设置", foreground='gray')
        # 切换图片时退出配准模式（保留已采集的点数据和缩略图）
        if self.registration_active:
            self._clear_registration_markers()
            self.registration_active = False
            self.registration_stage = ""
            self.reg_src_btn.state(['!pressed'])
            self.reg_dst_btn.state(['!pressed'])
        # 若有源图缩略图则保留（跨图配准需要）
        self.clear_current_bbox()
        self._refresh_tree()

        # 更新导航信息
        self._update_nav_label()

        # 搜索同名txt
        txt_path = self._get_txt_path()
        if os.path.exists(txt_path):
            self._load_annotations_from_file(txt_path)
            self._update_status(f"已加载已有标注: {os.path.basename(txt_path)}  ({len(self.annotations)} 条)")
        else:
            ow, oh = self.original_image_size
            dw, dh = self.pil_image.size
            if ow != dw:
                self._update_status(f"已加载图片: {os.path.basename(img_path)}  "
                                    f"显示 {dw}×{dh}  (原始 {ow}×{oh})")
            else:
                self._update_status(f"已加载图片: {os.path.basename(img_path)}  ({dw}×{dh})")

        # 重绘（含已保存标注框）
        self._redraw_all()

        # 自动投影（批量模式）
        if self.auto_project_enabled:
            self.root.after(100, self._try_auto_project)  # 延迟确保界面就绪

    def _get_txt_path(self) -> str:
        """获取当前图片对应的txt标注文件路径"""
        if self.current_image_idx < 0:
            return ""
        img_path = self.image_list[self.current_image_idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        save_dir = self.save_directory if self.save_directory else os.path.dirname(img_path)
        return os.path.join(save_dir, f"{base}.txt")

    # ══════════════════════════════════════════════════════════
    # 标注文件读写
    # ══════════════════════════════════════════════════════════
    def _save_annotations_to_file(self, filepath: str):
        """将当前标注写入txt文件（JSON对象，含work_region和annotations，bbox归一化0~1存储）"""
        if self.pil_image is None:
            return
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            iw, ih = self.original_image_size
            data = {
                "image_width": iw,
                "image_height": ih,
                "annotations": [ann.to_dict(iw, ih) for ann in self.annotations]
            }
            # 工作区域（归一化坐标）
            if self.work_region is not None:
                wr_x1, wr_y1, wr_x2, wr_y2 = self.work_region
                data["work_region"] = {
                    "x1": round(wr_x1 / iw, 6),
                    "y1": round(wr_y1 / ih, 6),
                    "x2": round(wr_x2 / iw, 6),
                    "y2": round(wr_y2 / ih, 6),
                }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("保存失败", f"无法写入标注文件:\n{filepath}\n\n{e}")

    def _load_annotations_from_file(self, filepath: str):
        """从txt文件加载标注（自动兼容旧数组格式和新对象格式，基于原始尺寸）"""
        self.annotations.clear()
        self.work_region = None
        if self.pil_image is None:
            return
        iw, ih = self.original_image_size
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                # 旧格式: 纯数组，只有标注
                for item in data:
                    ann = Annotation.from_dict(item, iw, ih)
                    if ann:
                        self.annotations.append(ann)
            elif isinstance(data, dict):
                # 新格式: {"annotations": [...], "work_region": {...}}
                ann_list = data.get("annotations", [])
                for item in ann_list:
                    ann = Annotation.from_dict(item, iw, ih)
                    if ann:
                        self.annotations.append(ann)

                # 加载工作区域
                wr_data = data.get("work_region")
                if wr_data:
                    wr_x1 = float(wr_data["x1"])
                    wr_y1 = float(wr_data["y1"])
                    wr_x2 = float(wr_data["x2"])
                    wr_y2 = float(wr_data["y2"])
                    # 检测格式：归一化(<=1) 或 像素(>1)
                    if max(wr_x1, wr_y1, wr_x2, wr_y2) <= 1.0:
                        wr_x1 = int(round(wr_x1 * iw))
                        wr_y1 = int(round(wr_y1 * ih))
                        wr_x2 = int(round(wr_x2 * iw))
                        wr_y2 = int(round(wr_y2 * ih))
                    else:
                        wr_x1 = int(round(wr_x1))
                        wr_y1 = int(round(wr_y1))
                        wr_x2 = int(round(wr_x2))
                        wr_y2 = int(round(wr_y2))
                    self.work_region = [wr_x1, wr_y1, wr_x2, wr_y2]
                    self.wr_status_label.config(
                        text=f"({wr_x1},{wr_y1})-({wr_x2},{wr_y2})", foreground='#00aa00')
                else:
                    self.wr_status_label.config(text="未设置", foreground='gray')
        except Exception as e:
            messagebox.showwarning("读取标注", f"读取标注文件出错:\n{filepath}\n\n{e}")
        self._refresh_tree()

    # ══════════════════════════════════════════════════════════
    # 图像显示 / 缩放
    # ══════════════════════════════════════════════════════════
    def _render_image(self):
        """根据当前缩放比例生成 tk_image 并更新 Canvas"""
        if self.pil_image is None:
            return

        w, h = self.pil_image.size
        new_w = max(1, int(w * self.zoom_scale))
        new_h = max(1, int(h * self.zoom_scale))

        # 使用高质量重采样
        resized = self.pil_image.resize((new_w, new_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)

        # 更新或创建Canvas图像
        if self.canvas_img_id is not None:
            self.canvas.itemconfig(self.canvas_img_id, image=self.tk_image)
        else:
            self.canvas_img_id = self.canvas.create_image(0, 0, anchor='nw',
                                                          image=self.tk_image)
        self.canvas.configure(scrollregion=(0, 0, new_w, new_h))

        # 更新缩放标签
        self.zoom_label.config(text=f"{int(self.zoom_scale * 100)}%")

    def _redraw_all(self, keep_scroll: bool = True):
        """重绘图像和所有标注框"""
        # 保存当前滚动位置（相对于图像的比例）
        sx_prop = sy_prop = 0.0
        if keep_scroll and self.canvas_img_id is not None:
            sx_prop = self.canvas.canvasx(0) / max(1, self.canvas.bbox('all')[2])
            sy_prop = self.canvas.canvasy(0) / max(1, self.canvas.bbox('all')[3])

        self.canvas.delete('all')
        self.canvas_img_id = None
        self.preview_rect_id = None
        self.saved_rect_ids.clear()

        if self.pil_image is None:
            return

        self._render_image()

        # 恢复滚动位置
        if keep_scroll:
            region = self.canvas.bbox('all')
            if region:
                self.canvas.xview_moveto(sx_prop)
                self.canvas.yview_moveto(sy_prop)

        # 绘制工作区域(绿色框 + 外部遮罩) — 在标注框下层
        self._draw_work_region()

        # 绘制配准点标记
        self._draw_registration_markers()

        # 绘制已保存的标注框（蓝色）
        for ann in self.annotations:
            c_x1, c_y1 = self._to_canvas(ann.x1, ann.y1)
            c_x2, c_y2 = self._to_canvas(ann.x2, ann.y2)
            rid = self.canvas.create_rectangle(
                c_x1, c_y1, c_x2, c_y2,
                outline='#00aaff', width=2, tags='saved_bbox'
            )
            # 在框左上角显示类别标签
            self.canvas.create_text(
                c_x1 + 2, c_y1 - 10 if c_y1 > 15 else c_y1 + 12,
                text=ann.category, anchor='w', fill='#00aaff',
                font=('Microsoft YaHei UI', 20, 'bold'), tags='saved_bbox'
            )
            self.saved_rect_ids.append(rid)

        # 绘制当前正在画的预览框（红色）
        if self.current_bbox is not None and self.current_category is not None:
            self._draw_preview_rect()

        # 确保图像在底层
        if self.canvas_img_id is not None:
            self.canvas.tag_lower(self.canvas_img_id)

    def _draw_preview_rect(self):
        """绘制当前预览红色框"""
        if self.preview_rect_id is not None:
            self.canvas.delete(self.preview_rect_id)
            # 同时删除预览标签
            self.canvas.delete('preview_label')

        if self.current_bbox is None or self.current_category is None:
            return

        c_x1, c_y1 = self._to_canvas(self.current_bbox[0], self.current_bbox[1])
        c_x2, c_y2 = self._to_canvas(self.current_bbox[2], self.current_bbox[3])
        self.preview_rect_id = self.canvas.create_rectangle(
            c_x1, c_y1, c_x2, c_y2,
            outline='red', width=2, dash=(4, 2)
        )
        self.canvas.create_text(
            c_x1 + 2, c_y1 - 10 if c_y1 > 15 else c_y1 + 12,
            text=self.current_category, anchor='w', fill='red',
            font=('Microsoft YaHei UI', 20, 'bold'), tags='preview_label'
        )

    def _update_bbox_display(self):
        """将当前编辑对象的坐标同步到微调输入框（标注框 / 工作区域 / 配准点）"""
        # ── 配准模式：显示最近配准点坐标 ──
        if self.registration_active:
            points = self.reg_points_src if self.registration_stage == "src" else self.reg_points_dst
            if points:
                px, py = points[-1]
                self.adjust_vars['x1'].set(px)
                self.adjust_vars['y1'].set(py)
                self.adjust_vars['x2'].set(px)
                self.adjust_vars['y2'].set(py)
            else:
                for k in ['x1', 'y1', 'x2', 'y2']:
                    self.adjust_vars[k].set(0)
            self.bbox_size_label.config(text="配准点坐标", foreground='#555')
            return

        # ── 更新尺寸显示 ──
        coords = None
        if self.work_region_adjust_mode and self.work_region is not None:
            coords = self.work_region
        elif self.current_bbox is not None:
            coords = self.current_bbox

        if coords is not None:
            for key, val in zip(['x1', 'y1', 'x2', 'y2'], coords):
                self.adjust_vars[key].set(val)
            w = coords[2] - coords[0]
            h = coords[3] - coords[1]
            area = w * h
            self.bbox_size_label.config(
                text=f"宽: {w}  高: {h}  面积: {area}",
                foreground='#007acc')
        else:
            for key in ['x1', 'y1', 'x2', 'y2']:
                self.adjust_vars[key].set(0)
            self.bbox_size_label.config(text="宽: —  高: —  面积: —", foreground='#555')

    def zoom_in(self):
        """放大"""
        if self.pil_image is None:
            return
        self.zoom_scale = min(self.zoom_scale * 1.25, 20.0)
        self._redraw_all()

    def zoom_out(self):
        """缩小"""
        if self.pil_image is None:
            return
        self.zoom_scale = max(self.zoom_scale / 1.25, 0.05)
        self._redraw_all()

    def zoom_100(self):
        """原始大小"""
        if self.pil_image is None:
            return
        self.zoom_scale = 1.0
        self._redraw_all()

    def fit_to_window(self, retry: int = 0):
        """适应窗口"""
        if self.pil_image is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if (cw < 10 or ch < 10) and retry < 10:
            # Canvas 尚未布局完成，延迟重试
            self.root.after(100, lambda: self.fit_to_window(retry + 1))
            return
        iw, ih = self.pil_image.size
        self.zoom_scale = min(cw / max(iw, 1), ch / max(ih, 1), 1.0)
        self._redraw_all(keep_scroll=False)

    # ══════════════════════════════════════════════════════════
    # 下采样（TIFF专用）
    # ══════════════════════════════════════════════════════════
    @staticmethod
    def _downsample_3x(img: Image.Image) -> Image.Image:
        """3倍stride下采样（取每第3个像素），用于多光谱TIFF"""
        arr = np.array(img)
        h, w = arr.shape[:2]
        crop_h, crop_w = (h // 3) * 3, (w // 3) * 3
        cropped = arr[:crop_h, :crop_w]
        if cropped.ndim == 3:
            down = cropped[0::3, 0::3, :]
        else:
            down = cropped[0::3, 0::3]
        return Image.fromarray(down)

    # ══════════════════════════════════════════════════════════
    # 坐标转换（图像始终位于Canvas坐标(0,0)，缩放原点即图像原点）
    # 标注存储使用原始图像坐标，显示时通过 display_scale 映射
    # ══════════════════════════════════════════════════════════
    def _to_canvas(self, img_x: int, img_y: int) -> Tuple[float, float]:
        """原始图像坐标 → 显示坐标 → Canvas坐标"""
        dx = img_x * self.display_scale
        dy = img_y * self.display_scale
        return (dx * self.zoom_scale, dy * self.zoom_scale)

    def _canvas_to_image(self, canvas_x: float, canvas_y: float) -> Tuple[int, int]:
        """Canvas坐标 → 显示坐标 → 原始图像坐标
        输入应为 canvas.canvasx/canvasy 转换后的Canvas坐标
        """
        dx = canvas_x / self.zoom_scale
        dy = canvas_y / self.zoom_scale
        return (int(round(dx / self.display_scale)),
                int(round(dy / self.display_scale)))

    # ══════════════════════════════════════════════════════════
    # 类别选择
    # ══════════════════════════════════════════════════════════
    def select_category(self, category: str):
        """选择标注类别"""
        self.current_category = category
        self.cat_hint.config(text=f"当前类别: {category}", foreground='#007acc')
        # 高亮选中的按钮
        for name, btn in self._cat_btn_ref.items():
            if name == category:
                btn.state(['pressed'])
            else:
                btn.state(['!pressed'])
        self._update_status(f"已选择类别「{category}」— 请在图像上拖拽画框")

    # ══════════════════════════════════════════════════════════
    # 鼠标画框交互
    # ══════════════════════════════════════════════════════════
    def _on_mouse_down(self, event):
        if self.pil_image is None:
            return
        # 配准点采集模式
        if self.registration_active:
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            ix, iy = self._canvas_to_image(cx, cy)
            # 钳位
            iw, ih = self.original_image_size
            ix = max(0, min(ix, iw))
            iy = max(0, min(iy, ih))
            if self.registration_stage == "src":
                self.reg_points_src.append((ix, iy))
            else:
                self.reg_points_dst.append((ix, iy))
            self._draw_registration_markers()
            self.reg_status_label.config(
                text=f"源:{len(self.reg_points_src)} 目标:{len(self.reg_points_dst)}",
                foreground='#cc9900')
            sn = len(self.reg_points_src) if self.registration_stage == "src" else len(self.reg_points_dst)
            self._update_status(f"已采集配准点 #{sn}: ({ix}, {iy})")
            return
        # 工作区域绘制模式（不需要选择类别）
        if self.drawing_work_region:
            self.drawing = True
            self.drag_start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
            self.canvas.configure(cursor='crosshair')
            return
        if self.current_category is None:
            self._update_status("请先选择一个标注类别")
            return
        # 开始画标注框时自动退出工作区域微调模式
        if self.work_region_adjust_mode:
            self.work_region_adjust_mode = False
            self.wr_adjust_var.set(False)
        # 记录起始点（Canvas坐标）
        self.drawing = True
        self.drag_start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.canvas.configure(cursor='crosshair')

    def _on_mouse_drag(self, event):
        if not self.drawing:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        sx, sy = self.drag_start

        # 计算图像坐标（绑定在原始图像范围内）
        img_w, img_h = self.original_image_size
        i_x1, i_y1 = self._canvas_to_image(sx, sy)
        i_x2, i_y2 = self._canvas_to_image(cx, cy)

        # 钳位到图像范围
        i_x1 = max(0, min(i_x1, img_w))
        i_y1 = max(0, min(i_y1, img_h))
        i_x2 = max(0, min(i_x2, img_w))
        i_y2 = max(0, min(i_y2, img_h))

        if self.drawing_work_region:
            # 工作区域绘制预览（轻量更新）
            self.work_region = [min(i_x1, i_x2), min(i_y1, i_y2),
                                max(i_x1, i_x2), max(i_y1, i_y2)]
            self._redraw_work_region()
            return

        self.current_bbox = [i_x1, i_y1, i_x2, i_y2]
        self._draw_preview_rect()
        self._update_bbox_display()

    def _on_mouse_up(self, event):
        if not self.drawing:
            return
        self.drawing = False
        self.canvas.configure(cursor='crosshair')

        # 工作区域绘制完成
        if self.drawing_work_region:
            self.drawing_work_region = False
            self.wr_toggle_btn.config(text="设置工作区域")
            if self.work_region is not None:
                wr_x1, wr_y1, wr_x2, wr_y2 = self.work_region
                if abs(wr_x2 - wr_x1) < 5 and abs(wr_y2 - wr_y1) < 5:
                    self.work_region = None
                    self.wr_status_label.config(text="未设置", foreground='gray')
                    self._redraw_all()
                    self._update_status("工作区域框太小，已忽略")
                    return
                self.wr_status_label.config(
                    text=f"({wr_x1},{wr_y1})-({wr_x2},{wr_y2})", foreground='#00aa00')
                # 自动进入微调模式
                self.work_region_adjust_mode = True
                self.wr_adjust_var.set(True)
                self._update_bbox_display()
                self._redraw_all()
                self._update_status(f"工作区域已设置: ({wr_x1},{wr_y1}) → ({wr_x2},{wr_y2}) — 微调面板已切换")
            return

        # 如果框太小（<3像素），视为无效
        if self.current_bbox is not None:
            x1, y1, x2, y2 = self.current_bbox
            if abs(x2 - x1) < 3 and abs(y2 - y1) < 3:
                self.clear_current_bbox()
                self._update_status("框太小，已忽略。请重新绘制。")
                return

        if self.current_bbox is not None:
            # 规范化为 x1<x2, y1<y2
            x1, y1, x2, y2 = self.current_bbox
            self.current_bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            self._draw_preview_rect()
            self._update_bbox_display()
            self._update_status(
                f"框已绘制: ({self.current_bbox[0]}, {self.current_bbox[1]}) "
                f"→ ({self.current_bbox[2]}, {self.current_bbox[3]}) — "
                f"可微调或点击「下一个」存储"
            )

    def _on_mouse_wheel(self, event):
        """鼠标滚轮缩放 (无Ctrl时也缩放，方便操作)"""
        if self.pil_image is None:
            return
        # Linux: Button-4/5, Windows/Mac: MouseWheel
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            self.zoom_in()
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            self.zoom_out()

    def _on_ctrl_wheel(self, event):
        """Ctrl+滚轮缩放（保留兼容）"""
        self._on_mouse_wheel(event)

    # ══════════════════════════════════════════════════════════
    # 微调框
    # ══════════════════════════════════════════════════════════
    def _start_repeat_timer(self, key: str, delta: int):
        """长按开始: 启动延迟定时器（单击由 command= 处理，不重复调用）"""
        self._stop_adjust_repeat()
        self._adjust_repeat_id = self.root.after(200,
                                                  lambda: self._adjust_repeat(key, delta))

    def _adjust_repeat(self, key: str, delta: int):
        """长按重复循环（间隔40ms）"""
        self.adjust_box(key, delta)
        self._adjust_repeat_id = self.root.after(40,
                                                  lambda: self._adjust_repeat(key, delta))

    def _stop_adjust_repeat(self, event=None):
        """停止长按重复"""
        if self._adjust_repeat_id is not None:
            self.root.after_cancel(self._adjust_repeat_id)
            self._adjust_repeat_id = None

    def adjust_box(self, key: str, delta: int):
        """微调框的某个参数（标注框 / 工作区域 / 配准点）"""
        if self.pil_image is None:
            return

        step = self.step_var.get() * delta
        idx_map = {'x1': 0, 'y1': 1, 'x2': 2, 'y2': 3}
        idx = idx_map[key]
        img_w, img_h = self.original_image_size
        max_vals = [img_w, img_h, img_w, img_h]

        # ── 配准模式：调整最近采集的配准点 ──
        if self.registration_active:
            points = self.reg_points_src if self.registration_stage == "src" else self.reg_points_dst
            if not points:
                self._update_status("请先在图像上点击采集配准点")
                return
            px, py = points[-1]
            if key in ('x1', 'x2'):
                px = max(0, min(img_w - 1, px + step))
            else:
                py = max(0, min(img_h - 1, py + step))
            points[-1] = (px, py)
            self._update_bbox_display()
            self._draw_registration_markers()
            sn = len(points)
            self._update_status(f"配准点 #{sn}: ({px}, {py}) [微调面板调整]")
            return

        # ── 工作区域微调模式 ──
        if self.work_region_adjust_mode and self.work_region is not None:
            # 调整工作区域
            new_val = self.work_region[idx] + step
            new_val = max(0, min(new_val, max_vals[idx]))
            if idx == 0:
                new_val = min(new_val, self.work_region[2] - 1)
            elif idx == 1:
                new_val = min(new_val, self.work_region[3] - 1)
            elif idx == 2:
                new_val = max(new_val, self.work_region[0] + 1)
            elif idx == 3:
                new_val = max(new_val, self.work_region[1] + 1)
            self.work_region[idx] = new_val
            self._update_bbox_display()
            self._redraw_all()
            self.wr_status_label.config(
                text=f"({self.work_region[0]},{self.work_region[1]})-"
                     f"({self.work_region[2]},{self.work_region[3]})",
                foreground='#00aa00')
            return

        if self.current_bbox is None:
            self._update_status("请先选择类别并画框，或勾选工作区域「微调模式」")
            return

        new_val = self.current_bbox[idx] + step
        new_val = max(0, min(new_val, max_vals[idx]))

        if idx == 0:  # x1
            new_val = min(new_val, self.current_bbox[2] - 1)
        elif idx == 1:  # y1
            new_val = min(new_val, self.current_bbox[3] - 1)
        elif idx == 2:  # x2
            new_val = max(new_val, self.current_bbox[0] + 1)
        elif idx == 3:  # y2
            new_val = max(new_val, self.current_bbox[1] + 1)

        self.current_bbox[idx] = new_val
        self._draw_preview_rect()
        self._update_bbox_display()

    def _on_entry_commit(self, key: str):
        """用户手动输入框参数后回车/失焦（标注框 / 工作区域 / 配准点）"""
        if self.pil_image is None:
            return
        try:
            val = self.adjust_vars[key].get()
        except tk.TclError:
            return
        idx_map = {'x1': 0, 'y1': 1, 'x2': 2, 'y2': 3}
        idx = idx_map[key]
        img_w, img_h = self.original_image_size
        max_vals = [img_w, img_h, img_w, img_h]
        val = max(0, min(val, max_vals[idx]))

        # ── 配准模式：手动输入配准点坐标 ──
        if self.registration_active:
            points = self.reg_points_src if self.registration_stage == "src" else self.reg_points_dst
            if not points:
                return
            px, py = points[-1]
            if key in ('x1', 'x2'):
                px = val
            else:
                py = val
            points[-1] = (px, py)
            self._update_bbox_display()
            self._draw_registration_markers()
            sn = len(points)
            self._update_status(f"配准点 #{sn}: ({px}, {py}) [手动输入]")
            return

        if self.work_region_adjust_mode and self.work_region is not None:
            # 调整工作区域
            if idx == 0:
                val = min(val, self.work_region[2] - 1)
            elif idx == 1:
                val = min(val, self.work_region[3] - 1)
            elif idx == 2:
                val = max(val, self.work_region[0] + 1)
            elif idx == 3:
                val = max(val, self.work_region[1] + 1)
            self.work_region[idx] = val
            self._update_bbox_display()
            self._redraw_all()
            self.wr_status_label.config(
                text=f"({self.work_region[0]},{self.work_region[1]})-"
                     f"({self.work_region[2]},{self.work_region[3]})",
                foreground='#00aa00')
            return

        if self.current_bbox is None:
            return

        if idx == 0:
            val = min(val, self.current_bbox[2] - 1)
        elif idx == 1:
            val = min(val, self.current_bbox[3] - 1)
        elif idx == 2:
            val = max(val, self.current_bbox[0] + 1)
        elif idx == 3:
            val = max(val, self.current_bbox[1] + 1)

        self.current_bbox[idx] = val
        self._draw_preview_rect()
        self._update_bbox_display()

    # ══════════════════════════════════════════════════════════
    # 标注操作
    # ══════════════════════════════════════════════════════════
    def next_annotation(self):
        """存储当前标注，准备下一个"""
        if self.current_bbox is None or self.current_category is None:
            self._update_status("请先选择类别并画框")
            return

        # 检查是否与已有标注重复
        x1, y1, x2, y2 = self.current_bbox
        for ann in self.annotations:
            if ann.category == self.current_category and \
               ann.x1 == x1 and ann.y1 == y1 and \
               ann.x2 == x2 and ann.y2 == y2:
                if not messagebox.askyesno("重复标注",
                                           f"类别「{self.current_category}」已存在完全相同的标注框。\n"
                                           "是否仍要添加？"):
                    return

        ann = Annotation(
            category=self.current_category,
            x1=x1, y1=y1, x2=x2, y2=y2
        )
        self.annotations.append(ann)
        self._refresh_tree()
        self._redraw_all()

        # 重置当前标注状态
        self.current_bbox = None
        self._update_bbox_display()
        self._draw_preview_rect()
        self._update_status(
            f"已存储「{ann.category}」标注 ({len(self.annotations)} 条) — "
            f"可继续选择类别标注"
        )

    def confirm_image(self):
        """确认当前图片标注完成，保存到txt"""
        if self.pil_image is None:
            return
        # 如果有未提交的当前框，询问是否一并保存
        if self.current_bbox is not None and self.current_category is not None:
            if messagebox.askyesno("未保存的标注",
                                   "当前还有未提交的标注框，是否一并保存？\n"
                                   f"类别: {self.current_category}\n"
                                   f"位置: {self.current_bbox}"):
                self.next_annotation()  # 先存入列表

        if not self.annotations and self.work_region is None:
            if not messagebox.askyesno("确认", "当前图片没有任何标注，确定保存空文件？"):
                return

        txt_path = self._get_txt_path()
        self._save_annotations_to_file(txt_path)
        self._update_status(f"✓ 已保存: {os.path.basename(txt_path)}  ({len(self.annotations)} 条标注)")
        messagebox.showinfo("保存成功",
                            f"标注已保存到:\n{txt_path}\n\n共 {len(self.annotations)} 条标注")

    def delete_annotation(self):
        """删除选中的标注"""
        selection = self.tree.selection()
        if not selection:
            self._update_status("请在标注结果列表中选中要删除的行")
            return
        item = selection[0]
        idx = int(self.tree.index(item))
        if 0 <= idx < len(self.annotations):
            ann = self.annotations[idx]
            if messagebox.askyesno("确认删除", f"确定删除标注:\n类别: {ann.category}\n"
                                              f"位置: ({ann.x1},{ann.y1}) → ({ann.x2},{ann.y2}) ?"):
                del self.annotations[idx]
                self._refresh_tree()
                self._redraw_all()
                self._update_status(f"已删除标注，当前共 {len(self.annotations)} 条")

    def clear_current_bbox(self):
        """清除当前正在画的预览框"""
        self.current_bbox = None
        self._update_bbox_display()
        self._draw_preview_rect()
        # 清除类别按钮选中状态
        for btn in self.cat_buttons:
            btn.state(['!pressed'])
        self.current_category = None
        self.cat_hint.config(text="当前类别: 未选择", foreground='#888')
        self._update_status("已撤销当前标注框")

    # ══════════════════════════════════════════════════════════
    # 有效工作区域
    # ══════════════════════════════════════════════════════════
    def _on_arrow(self, direction: str):
        """方向键: 配准模式下微调配准点，否则切换图片"""
        if self.registration_active:
            self._nudge_registration_point(direction)
            return
        if direction == 'left':
            self.prev_image()
        elif direction == 'right':
            self.next_image()

    def _nudge_registration_point(self, direction: str):
        """微调最近采集的配准点（1像素步长）"""
        points = self.reg_points_src if self.registration_stage == "src" else self.reg_points_dst
        if not points:
            return
        px, py = points[-1]
        iw, ih = self.original_image_size
        if direction == 'left':
            px = max(0, px - 1)
        elif direction == 'right':
            px = min(iw - 1, px + 1)
        elif direction == 'up':
            py = max(0, py - 1)
        elif direction == 'down':
            py = min(ih - 1, py + 1)
        points[-1] = (px, py)
        # 更新配准点计数显示
        sn = len(self.reg_points_src) if self.registration_stage == "src" else len(self.reg_points_dst)
        self._update_status(f"配准点 #{sn}: ({px}, {py}) [方向键微调]")
        self._draw_registration_markers()

    def _on_escape(self):
        """ESC: 取消配准模式 / 工作区域绘制 / 或撤销当前标注框"""
        if self.registration_active:
            self._stop_registration()
            return
        if self.drawing_work_region:
            self.drawing_work_region = False
            self.drawing = False
            self.wr_toggle_btn.config(text="设置工作区域")
            # 恢复之前的 work_region (如果有的话)
            self._draw_work_region()
            self._update_status("已取消工作区域绘制")
            return
        self.clear_current_bbox()

    def _toggle_work_region_mode(self):
        """进入/退出工作区域绘制模式"""
        if self.pil_image is None:
            return
        if self.drawing_work_region:
            # 取消绘制模式
            self.drawing_work_region = False
            self.drawing = False
            self.wr_toggle_btn.config(text="设置工作区域")
            self._update_status("已取消工作区域设置模式")
        else:
            # 进入绘制模式
            self.drawing_work_region = True
            self.wr_toggle_btn.config(text="绘制中... (ESC取消)")
            self._update_status("请在图像上拖拽绘制有效工作区域框 (ESC取消)")

    def _toggle_work_region_adjust(self):
        """切换工作区域微调模式"""
        self.work_region_adjust_mode = self.wr_adjust_var.get()
        if self.work_region_adjust_mode:
            if self.work_region is None:
                self.wr_adjust_var.set(False)
                self.work_region_adjust_mode = False
                self._update_status("请先设置工作区域")
                return
            self._update_bbox_display()
            self._update_status("微调面板当前控制: 工作区域 (绿色框)")
        else:
            self._update_bbox_display()
            self._update_status("微调面板当前控制: 标注框")

    def _clear_work_region(self):
        """清除当前工作区域"""
        if self.work_region is not None:
            self.work_region = None
            self.drawing_work_region = False
            self.drawing = False
            self.work_region_adjust_mode = False
            self.wr_adjust_var.set(False)
            self.wr_toggle_btn.config(text="设置工作区域")
            self.wr_status_label.config(text="未设置", foreground='gray')
            self._update_bbox_display()
            self._redraw_all()
            self._update_status("已清除工作区域")
        else:
            self._update_status("当前未设置工作区域")

    def _draw_work_region(self):
        """绘制工作区域(绿色框) — 由 _redraw_all 调用"""
        # 清除旧的工作区域元素
        if self.work_region_rect_id is not None:
            self.canvas.delete(self.work_region_rect_id)
            self.work_region_rect_id = None
        for rid in self.work_region_dim_ids:
            self.canvas.delete(rid)
        self.work_region_dim_ids.clear()
        self.canvas.delete('work_region_label')

        if self.work_region is None or self.pil_image is None:
            return

        wr_x1, wr_y1, wr_x2, wr_y2 = self.work_region
        c_x1, c_y1 = self._to_canvas(wr_x1, wr_y1)
        c_x2, c_y2 = self._to_canvas(wr_x2, wr_y2)

        # 四角标线 (L形角标，无填充，性能好)
        corner_len = 20
        for (cx, cy, dx, dy) in [
            (c_x1, c_y1,  1,  1),  # 左上
            (c_x2, c_y1, -1,  1),  # 右上
            (c_x1, c_y2,  1, -1),  # 左下
            (c_x2, c_y2, -1, -1),  # 右下
        ]:
            rid1 = self.canvas.create_line(cx, cy, cx + dx * corner_len, cy,
                                           fill='#00ff00', width=3, tags='work_region')
            rid2 = self.canvas.create_line(cx, cy, cx, cy + dy * corner_len,
                                           fill='#00ff00', width=3, tags='work_region')
            self.work_region_dim_ids.extend([rid1, rid2])

        # 工作区域绿色边框(虚线，区别于标注框)
        self.work_region_rect_id = self.canvas.create_rectangle(
            c_x1, c_y1, c_x2, c_y2,
            outline='#00ff00', width=2, dash=(8, 4), tags='work_region'
        )
        # 标签
        self.canvas.create_text(
            c_x1 + 4, c_y1 - 14 if c_y1 > 18 else c_y1 + 16,
            text='工作区域', anchor='w', fill='#00ff00',
            font=('Microsoft YaHei UI', 20, 'bold'), tags='work_region_label'
        )

    def _redraw_work_region(self):
        """轻量更新工作区域视觉(拖拽时使用，不重绘整个 canvas)"""
        if self.work_region_rect_id is not None:
            self.canvas.delete(self.work_region_rect_id)
            self.work_region_rect_id = None
        for rid in self.work_region_dim_ids:
            self.canvas.delete(rid)
        self.work_region_dim_ids.clear()
        self.canvas.delete('work_region_label')

        if self.work_region is None:
            return

        wr_x1, wr_y1, wr_x2, wr_y2 = self.work_region
        c_x1, c_y1 = self._to_canvas(wr_x1, wr_y1)
        c_x2, c_y2 = self._to_canvas(wr_x2, wr_y2)

        corner_len = 20
        for (cx, cy, dx, dy) in [
            (c_x1, c_y1,  1,  1),
            (c_x2, c_y1, -1,  1),
            (c_x1, c_y2,  1, -1),
            (c_x2, c_y2, -1, -1),
        ]:
            rid1 = self.canvas.create_line(cx, cy, cx + dx * corner_len, cy,
                                           fill='#00ff00', width=3, tags='work_region')
            rid2 = self.canvas.create_line(cx, cy, cx, cy + dy * corner_len,
                                           fill='#00ff00', width=3, tags='work_region')
            self.work_region_dim_ids.extend([rid1, rid2])

        self.work_region_rect_id = self.canvas.create_rectangle(
            c_x1, c_y1, c_x2, c_y2,
            outline='#00ff00', width=2, dash=(8, 4), tags='work_region'
        )
        self.canvas.create_text(
            c_x1 + 4, c_y1 - 14 if c_y1 > 18 else c_y1 + 16,
            text='工作区域', anchor='w', fill='#00ff00',
            font=('Microsoft YaHei UI', 20, 'bold'), tags='work_region_label'
        )

    # ══════════════════════════════════════════════════════════
    # 配准变换 (双相机投影)
    # ══════════════════════════════════════════════════════════
    def _start_registration(self, stage: str):
        """进入配准点采集模式 (stage='src' or 'dst')"""
        if self.pil_image is None:
            return
        if self.registration_active and self.registration_stage == stage:
            # 再次点击同一按钮 → 退出
            self._stop_registration()
            return
        # 退出其他模式
        if self.drawing_work_region:
            self.drawing_work_region = False
            self.drawing = False
            self.wr_toggle_btn.config(text="设置工作区域")
        # 进入配准模式
        self.registration_active = True
        self.registration_stage = stage
        # 高亮按钮
        if stage == "src":
            self.reg_src_btn.state(['pressed'])
            self.reg_dst_btn.state(['!pressed'])
        else:
            self.reg_dst_btn.state(['pressed'])
            self.reg_src_btn.state(['!pressed'])
        self._update_status(f"配准模式: 请在图像上点击特征点 ({'源图' if stage == 'src' else '目标图'}) — 再次点击按钮退出")
        self._redraw_all()  # 触发重绘以显示配准点标记和缩略图

    def _stop_registration(self):
        """退出配准点采集模式"""
        # 退出源图采点时，创建缩略图供目标图采点参考
        if self.registration_stage == "src" and self.reg_points_src:
            self._create_registration_thumbnail()
        self.registration_active = False
        self.registration_stage = ""
        self.reg_src_btn.state(['!pressed'])
        self.reg_dst_btn.state(['!pressed'])
        self._redraw_all()
        self._update_status("已退出配准模式")

    def _undo_last_registration_point(self):
        """撤销最近采集的一个配准点（优先撤销当前活跃模式的点）"""
        if self.registration_active:
            points = self.reg_points_src if self.registration_stage == "src" else self.reg_points_dst
        else:
            # 未活跃时，优先撤销目标图点（通常是后采集的）
            points = self.reg_points_dst if self.reg_points_dst else self.reg_points_src
        if not points:
            self._update_status("没有可撤销的配准点")
            return
        removed = points.pop()
        self._draw_registration_markers()
        if self.registration_active:
            self._update_bbox_display()
        self.reg_status_label.config(
            text=f"源:{len(self.reg_points_src)} 目标:{len(self.reg_points_dst)}",
            foreground='#cc9900' if (self.reg_points_src or self.reg_points_dst) else 'gray')
        self._update_status(f"已撤销配准点: ({removed[0]}, {removed[1]})")
        # 更新源图缩略图（源图点变化时）
        if self.reg_points_src:
            self._create_registration_thumbnail()
        else:
            self.reg_source_thumbnail = None

    def _clear_registration(self):
        """清除所有配准点数据"""
        self.reg_points_src.clear()
        self.reg_points_dst.clear()
        self.reg_source_thumbnail = None
        self._clear_registration_markers()
        self.reg_status_label.config(text="源:0 目标:0", foreground='gray')
        self._update_status("已清除所有配准点")

    def _clear_registration_markers(self):
        """清除配准点 Canvas 标记"""
        for rid in self.reg_marker_ids:
            self.canvas.delete(rid)
        self.reg_marker_ids.clear()
        self.canvas.delete('reg_marker')
        # 清除缩略图
        if self.reg_thumb_id is not None:
            self.canvas.delete(self.reg_thumb_id)
            self.reg_thumb_id = None

    def _create_registration_thumbnail(self):
        """为源图创建含配准点标记的缩略图，供目标图采点时参考"""
        from PIL import ImageDraw
        if self.pil_image is None or not self.reg_points_src:
            self.reg_source_thumbnail = None
            return
        try:
            max_w, max_h = 280, 210
            thumb = self.pil_image.copy()
            thumb.thumbnail((max_w, max_h), Image.LANCZOS)
            tw, th = thumb.size
            draw = ImageDraw.Draw(thumb)
            # 缩放比例: 显示图像 → 缩略图
            dw, dh = self.pil_image.size
            sx = tw / dw
            sy = th / dh
            for i, (px, py) in enumerate(self.reg_points_src):
                dx = px * self.display_scale
                dy = py * self.display_scale
                tx, ty = dx * sx, dy * sy
                r = 5
                draw.ellipse([tx - r, ty - r, tx + r, ty + r],
                             fill='yellow', outline='#cc9900', width=1)
                draw.text((tx + 7, ty - 8), str(i + 1), fill='white')
            self.reg_source_thumbnail = ImageTk.PhotoImage(thumb)
        except Exception:
            self.reg_source_thumbnail = None

    def _draw_registration_markers(self):
        """绘制配准点标记 (带编号的黄色圆形+十字) 以及源图参考缩略图"""
        self._clear_registration_markers()
        if not self.registration_active:
            return

        points = self.reg_points_src if self.registration_stage == "src" else self.reg_points_dst
        if not points:
            return

        for i, (px, py) in enumerate(points):
            cx, cy = self._to_canvas(px, py)
            r = 12
            # 黄色填充圆
            rid = self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                          fill='#ffff00', outline='#cc9900',
                                          width=2, tags='reg_marker')
            self.reg_marker_ids.append(rid)
            # 编号文字
            rid = self.canvas.create_text(cx, cy, text=str(i + 1),
                                          fill='#000000',
                                          font=('Microsoft YaHei UI', 16, 'bold'),
                                          tags='reg_marker')
            self.reg_marker_ids.append(rid)
            # 十字线
            rid = self.canvas.create_line(cx - 16, cy, cx + 16, cy,
                                          fill='#cc9900', width=1, tags='reg_marker')
            self.reg_marker_ids.append(rid)
            rid = self.canvas.create_line(cx, cy - 16, cx, cy + 16,
                                          fill='#cc9900', width=1, tags='reg_marker')
            self.reg_marker_ids.append(rid)

        # 目标图采点时，显示源图缩略图作为参考
        if self.registration_stage == "dst" and self.reg_source_thumbnail is not None:
            tw = self.reg_source_thumbnail.width()
            th = self.reg_source_thumbnail.height()
            pad = 10
            # 放在 Canvas 右下角
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw < 100:
                cw = 800
            if ch < 100:
                ch = 600
            x0, y0 = cw - tw - pad, ch - th - pad
            # 半透明背景
            rid = self.canvas.create_rectangle(
                x0 - 3, y0 - 22, x0 + tw + 3, y0 + th + 3,
                fill='#1a1a1a', outline='#cc9900', width=2, tags='reg_marker')
            self.reg_marker_ids.append(rid)
            self.reg_thumb_id = self.canvas.create_image(
                x0, y0, anchor='nw', image=self.reg_source_thumbnail,
                tags='reg_marker')
            rid = self.canvas.create_text(
                x0 + 4, y0 - 8, text='源图参考 (按序点击对应点)',
                anchor='w', fill='#cc9900',
                font=('Microsoft YaHei UI', 14, 'bold'), tags='reg_marker')
            self.reg_marker_ids.append(rid)

    @staticmethod
    def _compute_homography_dlt(src_pts, dst_pts):
        """DLT算法计算3x3单应性矩阵 (需要>=4对点)"""
        A = []
        for (x, y), (xp, yp) in zip(src_pts, dst_pts):
            A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
            A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
        A = np.array(A, dtype=np.float64)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        H = H / H[2, 2]
        return H

    def _compute_and_save_homography(self):
        """计算单应性矩阵并保存到JSON文件"""
        if len(self.reg_points_src) < 4 or len(self.reg_points_dst) < 4:
            messagebox.showwarning("点数不足",
                                   f"源图和目标图各需要至少4个配准点。\n"
                                   f"当前: 源图 {len(self.reg_points_src)} 点, "
                                   f"目标图 {len(self.reg_points_dst)} 点")
            return
        if len(self.reg_points_src) != len(self.reg_points_dst):
            messagebox.showwarning("点数不匹配",
                                   f"源图({len(self.reg_points_src)}点)和"
                                   f"目标图({len(self.reg_points_dst)}点)点数不一致")
            return

        try:
            H = self._compute_homography_dlt(self.reg_points_src, self.reg_points_dst)
            self.homography_matrix = H
        except Exception as e:
            messagebox.showerror("计算失败", f"单应性矩阵计算失败:\n{e}")
            return

        # 保存
        filepath = filedialog.asksaveasfilename(
            title="保存单应性矩阵",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="homography.json"
        )
        if not filepath:
            return

        try:
            data = {
                "matrix": H.tolist(),
            }
            if self.pil_image:
                data["dst_resolution"] = list(self.original_image_size)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            self.homography_file = filepath
            self.reg_matrix_label.config(
                text=f"矩阵: {H[0,0]:.4f} {H[0,1]:.4f} ...", foreground='#00aa00')
            self._update_status(f"单应性矩阵已保存: {os.path.basename(filepath)}")
            messagebox.showinfo("成功",
                                f"单应性矩阵已保存到:\n{filepath}\n\n"
                                f"矩阵:\n{H[0]}\n{H[1]}\n{H[2]}")
        except Exception as e:
            messagebox.showerror("保存失败", f"无法保存矩阵文件:\n{e}")

    def _load_homography_from_file(self):
        """从JSON文件加载单应性矩阵"""
        filepath = filedialog.askopenfilename(
            title="加载单应性矩阵",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            H = np.array(data["matrix"], dtype=np.float64)
            if H.shape != (3, 3):
                raise ValueError("矩阵形状不是3x3")
            self.homography_matrix = H
            self.homography_file = filepath
            self.reg_matrix_label.config(
                text=f"矩阵: {H[0,0]:.4f} {H[0,1]:.4f} ...", foreground='#00aa00')
            self._update_status(f"已加载单应性矩阵: {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载矩阵文件:\n{e}")

    @staticmethod
    def _transform_point(x: float, y: float, H: np.ndarray) -> Tuple[float, float]:
        """将点(x,y)通过3x3单应性矩阵H变换"""
        p = np.array([x, y, 1.0])
        pp = H @ p
        return (pp[0] / pp[2], pp[1] / pp[2])

    @staticmethod
    def _transform_bbox(x1: int, y1: int, x2: int, y2: int,
                        H: np.ndarray) -> Tuple[int, int, int, int]:
        """将bbox四个角点通过单应性变换，返回轴对齐包围盒(AABB)"""
        corners = [
            FODAnnotationTool._transform_point(x1, y1, H),
            FODAnnotationTool._transform_point(x2, y1, H),
            FODAnnotationTool._transform_point(x1, y2, H),
            FODAnnotationTool._transform_point(x2, y2, H),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return (int(round(min(xs))), int(round(min(ys))),
                int(round(max(xs))), int(round(max(ys))))

    # ══════════════════════════════════════════════════════════
    # 投影标注（核心逻辑 + 手动选择 + 批量自动）
    # ══════════════════════════════════════════════════════════
    def _project_from_txt_file(self, src_txt: str, silent: bool = False) -> int:
        """从指定txt文件加载标注、变换并导入到当前图片。返回投影条数，失败返回-1"""
        if self.pil_image is None:
            return -1

        # 加载源标注
        try:
            with open(src_txt, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            if not silent:
                messagebox.showerror("读取失败", f"无法读取源标注文件:\n{src_txt}\n\n{e}")
            return -1

        # 解析标注 (兼容新旧格式)
        if isinstance(data, list):
            ann_list = data
        elif isinstance(data, dict):
            ann_list = data.get("annotations", [])
        else:
            if not silent:
                messagebox.showerror("格式错误", "无法识别标注文件格式")
            return -1

        # 检查是否有工作区域可投影
        has_work_region = isinstance(data, dict) and "work_region" in data
        if not ann_list and not has_work_region:
            return 0

        # 获取源图尺寸
        if isinstance(data, dict):
            src_w = data.get("image_width", 0)
            src_h = data.get("image_height", 0)
        else:
            src_w, src_h = 0, 0

        # 检查坐标格式（无标注框时默认为归一化格式）
        is_normalized = True
        if ann_list:
            first_bbox = ann_list[0].get("bbox", {})
            is_normalized = max(float(first_bbox.get("x1", 0)), float(first_bbox.get("y1", 0)),
                                float(first_bbox.get("x2", 0)), float(first_bbox.get("y2", 0))) <= 1.0

        if is_normalized:
            if src_w <= 0 or src_h <= 0:
                from tkinter import simpledialog
                src_w = simpledialog.askinteger("源图尺寸", "源图宽度(像素):",
                                                minvalue=1, maxvalue=99999)
                if src_w is None:
                    return -1
                src_h = simpledialog.askinteger("源图尺寸", "源图高度(像素):",
                                                minvalue=1, maxvalue=99999)
                if src_h is None:
                    return -1
        else:
            src_w, src_h = 1, 1

        H = self.homography_matrix

        # 变换每个标注
        projected_count = 0
        for item in ann_list:
            try:
                bbox = item["bbox"]
                if is_normalized:
                    bx1 = float(bbox["x1"]) * src_w
                    by1 = float(bbox["y1"]) * src_h
                    bx2 = float(bbox["x2"]) * src_w
                    by2 = float(bbox["y2"]) * src_h
                else:
                    bx1 = float(bbox["x1"])
                    by1 = float(bbox["y1"])
                    bx2 = float(bbox["x2"])
                    by2 = float(bbox["y2"])

                nx1, ny1, nx2, ny2 = self._transform_bbox(
                    int(round(bx1)), int(round(by1)),
                    int(round(bx2)), int(round(by2)), H)

                iw, ih = self.original_image_size
                nx1 = max(0, min(nx1, iw))
                ny1 = max(0, min(ny1, ih))
                nx2 = max(0, min(nx2, iw))
                ny2 = max(0, min(ny2, ih))

                if abs(nx2 - nx1) < 3 or abs(ny2 - ny1) < 3:
                    continue

                category = ID_TO_CATEGORY.get(int(item["id"]),
                                              item.get("name", f"未知{int(item['id'])}"))
                ann = Annotation(category=category,
                                 x1=nx1, y1=ny1, x2=nx2, y2=ny2)
                self.annotations.append(ann)
                projected_count += 1
            except (KeyError, ValueError, TypeError):
                continue

        # ── 投影源图的工作区域 ──
        wr_projected = False
        if isinstance(data, dict) and "work_region" in data:
            wr = data["work_region"]
            try:
                if is_normalized:
                    wx1 = float(wr["x1"]) * src_w
                    wy1 = float(wr["y1"]) * src_h
                    wx2 = float(wr["x2"]) * src_w
                    wy2 = float(wr["y2"]) * src_h
                else:
                    wx1 = float(wr["x1"])
                    wy1 = float(wr["y1"])
                    wx2 = float(wr["x2"])
                    wy2 = float(wr["y2"])

                nwx1, nwy1, nwx2, nwy2 = self._transform_bbox(
                    int(round(wx1)), int(round(wy1)),
                    int(round(wx2)), int(round(wy2)), H)

                iw, ih = self.original_image_size
                nwx1 = max(0, min(nwx1, iw))
                nwy1 = max(0, min(nwy1, ih))
                nwx2 = max(0, min(nwx2, iw))
                nwy2 = max(0, min(nwy2, ih))

                if abs(nwx2 - nwx1) >= 5 and abs(nwy2 - nwy1) >= 5:
                    self.work_region = [nwx1, nwy1, nwx2, nwy2]
                    self.wr_status_label.config(
                        text=f"({nwx1},{nwy1})-({nwx2},{nwy2})", foreground='#00aa00')
                    wr_projected = True
            except (KeyError, ValueError, TypeError):
                pass

        self._refresh_tree()
        self._redraw_all()
        txt_name = os.path.basename(src_txt)
        wr_msg = " + 工作区域" if wr_projected else ""
        self._update_status(
            f"已从源图投影 {projected_count} 条标注{wr_msg} ({txt_name})")

        if not silent:
            messagebox.showinfo("投影完成",
                                f"成功投影 {projected_count}/{len(ann_list)} 条标注{wr_msg}\n"
                                f"来源: {txt_name}\n\n"
                                f"请检查并微调后保存。")
        return projected_count

    def _project_annotations_from_source(self):
        """手动选择源标注txt → 变换 → 导入到当前图片"""
        if self.homography_matrix is None:
            if not messagebox.askyesno("无变换矩阵",
                                       "尚未设置单应性矩阵。是否先加载已有矩阵文件？"):
                return
            self._load_homography_from_file()
            if self.homography_matrix is None:
                return

        if self.pil_image is None:
            self._update_status("请先打开目标图片")
            return

        # 如果有源标注文件夹，默认定位到该文件夹
        initial_dir = self.source_annotation_folder if self.source_annotation_folder else ""
        src_txt = filedialog.askopenfilename(
            title="选择源图标注文件 (txt)",
            initialdir=initial_dir,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not src_txt:
            return

        self._project_from_txt_file(src_txt)

    def _select_source_annotation_folder(self):
        """选择源标注文件夹（批量投影模式）"""
        folder = filedialog.askdirectory(title="选择源标注文件夹（含txt文件）")
        if not folder:
            return
        self.source_annotation_folder = folder
        self._update_source_folder_display()
        # 立即尝试投影当前图片
        if self.auto_project_enabled:
            self._try_auto_project()

    def _toggle_auto_project(self):
        """切换自动投影开关"""
        self.auto_project_enabled = self.auto_project_var.get()
        if self.auto_project_enabled:
            if not self.source_annotation_folder:
                self.auto_project_var.set(False)
                self.auto_project_enabled = False
                self._update_status("请先选择源标注文件夹")
                return
            self._update_status("自动投影已开启 — 切换图片时自动匹配源标注")
            self._try_auto_project()
        else:
            self._update_status("自动投影已关闭")

    def _try_auto_project(self):
        """根据当前目标图片文件名，自动匹配源标注txt并投影"""
        if not self.auto_project_enabled or not self.source_annotation_folder:
            return
        if self.homography_matrix is None or self.pil_image is None:
            return
        if not self.image_list:
            return

        current_img = self.image_list[self.current_image_idx]
        base = os.path.splitext(os.path.basename(current_img))[0]
        src_txt = os.path.join(self.source_annotation_folder, f"{base}.txt")

        self._update_source_folder_display(base)

        if os.path.exists(src_txt):
            # 检查是否已有标注（避免重复投影覆盖已修改的标注）
            if self.annotations:
                if not messagebox.askyesno("已有标注",
                                           f"当前图片已有 {len(self.annotations)} 条标注。\n"
                                           f"自动投影将覆盖现有标注，是否继续？"):
                    return
                self.annotations.clear()
            result = self._project_from_txt_file(src_txt, silent=True)
            if result > 0:
                self._update_status(
                    f"已自动投影 {result} 条标注 ({base}.txt)")
        else:
            # 检查是否已经是源文件夹中的最后一个txt
            all_txts = sorted([f for f in os.listdir(self.source_annotation_folder)
                              if f.endswith('.txt')])
            if all_txts:
                last_base = os.path.splitext(all_txts[-1])[0]
                # 按字母序比较当前文件名与最后一个txt
                if base > last_base:
                    messagebox.showinfo("序列结束",
                                        f"源标注文件夹中已无更多匹配的txt文件。\n"
                                        f"最后一个txt: {all_txts[-1]}\n"
                                        f"当前图片: {os.path.basename(current_img)}")
            self._update_status(f"未找到匹配的源标注: {base}.txt")

    def _update_source_folder_display(self, current_base: str = ""):
        """更新源文件夹状态显示"""
        if not self.source_annotation_folder:
            self.reg_folder_status.config(text="源文件夹: 未选择", foreground='gray')
            return
        folder_short = self.source_annotation_folder
        if len(folder_short) > 30:
            folder_short = "..." + folder_short[-27:]
        if current_base:
            self.reg_folder_status.config(
                text=f"{folder_short}  |  {current_base}.txt",
                foreground='#007acc')
        else:
            self.reg_folder_status.config(
                text=f"源文件夹: {folder_short}", foreground='#555')

    def _refresh_tree(self):
        """刷新Treeview显示"""
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, ann in enumerate(self.annotations, 1):
            cid = CATEGORY_TO_ID.get(ann.category, -1)
            self.tree.insert('', 'end', values=(i, cid, ann.category, ann.x1, ann.y1, ann.x2, ann.y2))

    def _on_tree_double_click(self, event):
        """双击Treeview行: 加载该标注到画布进行修改"""
        selection = self.tree.selection()
        if not selection:
            return
        item = selection[0]
        idx = int(self.tree.index(item))
        if 0 <= idx < len(self.annotations):
            ann = self.annotations[idx]
            # 将标注加载为当前编辑框
            self.current_category = ann.category
            self.cat_hint.config(text=f"当前类别: {ann.category}", foreground='#007acc')
            # 高亮按钮
            for name, btn in self._cat_btn_ref.items():
                if name == ann.category:
                    btn.state(['pressed'])
                else:
                    btn.state(['!pressed'])
            self.current_bbox = [ann.x1, ann.y1, ann.x2, ann.y2]
            self._draw_preview_rect()
            self._update_bbox_display()
            # 从列表中移除（修改后需重新"下一个"提交）
            del self.annotations[idx]
            self._refresh_tree()
            self._redraw_all()
            self._update_status(
                f"正在修改标注「{ann.category}」— 修改后请点击「下一个」重新存储"
            )

    # ══════════════════════════════════════════════════════════
    # 下采样切换
    # ══════════════════════════════════════════════════════════
    def _toggle_downsample(self):
        """切换3倍下采样模式，重新加载当前图像"""
        self.downsample_enabled = self.downsample_var.get()
        if self.pil_image is None:
            return
        # 询问是否保存当前标注
        if self.annotations and self.current_bbox is not None:
            if messagebox.askyesno("切换模式", "切换下采样模式将丢失未提交的标注框，是否继续？"):
                pass
            else:
                self.downsample_var.set(not self.downsample_enabled)
                return
        self.clear_current_bbox()
        self._load_and_display()

    # ══════════════════════════════════════════════════════════
    # 导航
    # ══════════════════════════════════════════════════════════
    def prev_image(self):
        """上一张图片"""
        if not self.image_list:
            return
        if self.current_image_idx <= 0:
            self._update_status("已经是第一张")
            return
        self._auto_save_if_needed()
        self.current_image_idx -= 1
        self._load_and_display()

    def next_image(self):
        """下一张图片"""
        if not self.image_list:
            return
        if self.current_image_idx >= len(self.image_list) - 1:
            self._update_status("已经是最后一张")
            return
        self._auto_save_if_needed()
        self.current_image_idx += 1
        self._load_and_display()

    def _auto_save_if_needed(self):
        """切换图片前自动保存（如果标注已通过「确定」保存则跳过，否则询问并保存）"""
        if self.pil_image is None:
            return
        # 如果有未提交的当前框，提示
        if self.current_bbox is not None and self.current_category is not None:
            if messagebox.askyesno("未保存的标注",
                                   "当前有未提交的标注框，切换前是否保存？"):
                self.next_annotation()

        if self.annotations or self.work_region is not None:
            txt_path = self._get_txt_path()
            if not os.path.exists(txt_path):
                if messagebox.askyesno("自动保存",
                                       f"当前图片有 {len(self.annotations)} 条标注"
                                       f"{' + 工作区域' if self.work_region else ''}，"
                                       "是否保存后再切换？"):
                    self._save_annotations_to_file(txt_path)
                    self._update_status(f"已自动保存: {os.path.basename(txt_path)}")
            else:
                # txt已存在（可能是加载的或之前保存的），检查内容是否一致
                self._save_annotations_to_file(txt_path)

    def _update_nav_label(self):
        """更新导航标签和图片下拉列表"""
        if not self.image_list:
            self.pos_label.config(text="未加载图片")
            self.image_combo['values'] = []
            self._combo_var.set("")
            return
        total = len(self.image_list)
        cur = self.current_image_idx + 1
        fname = os.path.basename(self.image_list[self.current_image_idx])
        self.pos_label.config(text=f"[{cur}/{total}]")
        # 更新下拉列表（每次切换图片/文件夹均重建，保证内容一致）
        self.image_combo['values'] = [os.path.basename(p) for p in self.image_list]
        self._combo_var.set(fname)

    def _on_combo_select(self, event=None):
        """下拉框选择图片后跳转"""
        selected = self._combo_var.get()
        if not selected or not self.image_list:
            return
        # 按文件名查找索引
        for i, path in enumerate(self.image_list):
            if os.path.basename(path) == selected:
                if i == self.current_image_idx:
                    return  # 同一张图，无需跳转
                self._auto_save_if_needed()
                self.current_image_idx = i
                self._load_and_display()
                return
        self._update_status(f"未找到图片: {selected}")

    # ══════════════════════════════════════════════════════════
    # 保存目录
    # ══════════════════════════════════════════════════════════
    def select_save_dir(self):
        """选择自定义保存目录，自动加载新目录下的标注"""
        d = filedialog.askdirectory(title="选择标注保存目录")
        if d:
            self.save_directory = d
            self.save_dir_label.config(text=d, foreground='#000')
        else:
            self.save_directory = ""
            self.save_dir_label.config(text="(图片同目录)", foreground='gray')
        # 自动刷新：从新保存目录加载标注
        if self.pil_image is not None:
            txt_path = self._get_txt_path()
            if os.path.exists(txt_path):
                self._load_annotations_from_file(txt_path)
                self._redraw_all()
                self._update_status(f"已从新目录加载标注: {os.path.basename(txt_path)}  ({len(self.annotations)} 条)")
            else:
                self._update_status(f"保存目录已更改 (新目录中无已有标注)")

    # ══════════════════════════════════════════════════════════
    # 杂项
    # ══════════════════════════════════════════════════════════
    def _update_status(self, text: str):
        """更新状态栏"""
        self.status_var.set(text)

    def _show_about(self):
        messagebox.showinfo("关于",
                            "FOD 图像标注工具 v1.2\n\n"
                            "用于外来物碎片(FOD)图像的 Bounding Box 标注。\n"
                            "支持12种类别（金属/橡胶/塑料/木头/纸板/\n"
                            "织物/动物/玻璃/石头/泡沫/油渍/水），\n"
                            "缩放标注，结果存为JSON结构体格式txt。\n"
                            "类别以0-11整数编号存储，{}分级嵌套。\n\n"
                            "快捷键:\n"
                            "  左右箭头: 切换图片\n"
                            "  Ctrl+/-: 缩放\n"
                            "  Esc: 撤销当前框\n"
                            "  滚轮: 缩放")

    def _on_close(self):
        """关闭窗口时的处理"""
        if self.annotations or self.work_region is not None:
            if messagebox.askyesno("退出确认",
                                   "当前图片有未保存的标注，是否保存后退出？"):
                txt_path = self._get_txt_path()
                # 先提交未保存的当前框
                if self.current_bbox is not None and self.current_category is not None:
                    if messagebox.askyesno("未保存的标注",
                                           "还有未提交的标注框，是否一并保存？"):
                        self.next_annotation()
                self._save_annotations_to_file(txt_path)
        self.root.destroy()


# ══════════════════════════════════════════════════════════════
# 启动入口
# ══════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()

    # 设置DPI感知 (Windows高分屏)
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    # ── 全局字体放大 ──
    FONT_BASE = ('Microsoft YaHei UI', 20)
    FONT_BOLD = ('Microsoft YaHei UI', 20, 'bold')
    FONT_SMALL = ('Microsoft YaHei UI', 19)
    FONT_MENU = ('Microsoft YaHei UI', 23)       # 菜单/对话框 +6

    # 设置 tk 根默认字体（影响对话框等）
    root.option_add('*Font', FONT_MENU)
    root.option_add('*Dialog.msg.font', FONT_MENU)

    style = ttk.Style()
    available = style.theme_names()
    for preferred in ('vista', 'clam', 'alt', 'default'):
        if preferred in available:
            style.theme_use(preferred)
            break

    style.configure('TButton', font=FONT_BASE)
    style.configure('TCheckbutton', font=FONT_BASE)
    style.configure('TLabel', font=FONT_BASE)
    style.configure('TLabelframe.Label', font=FONT_BOLD)
    style.configure('TEntry', font=FONT_BASE)
    style.configure('TSpinbox', font=FONT_BASE)
    style.configure('Treeview', font=FONT_BASE, rowheight=48)
    style.configure('Treeview.Heading', font=FONT_BOLD)
    # 菜单字体 +6
    style.configure('TMenu', font=FONT_MENU)
    style.configure('TStatusbar.TLabel', font=FONT_SMALL)

    app = FODAnnotationTool(root)
    root.mainloop()


if __name__ == '__main__':
    main()
